import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import math


class SyngistixCalibration:
    """
    Uses Measurement Objects for calibration. Spotsizes are either tuples (x, y, z) or (d, z), where x, y and d are in micrometers and z in nanometers.
    """

    def __init__(self, standards: list[tuple], spotsizes: list, film_thickness: float, film_concentration_percent: float):
        self.__standards = standards
        self.__spotsizes = spotsizes
        self.__film_thickness = film_thickness
        self.__film_concentration_percent = film_concentration_percent

        self.calibration = dict()
        self.analyte_mass = dict()
        self.analyte_intensity = dict()

    def merge_peak_df(self, measurements: list, isotope: str):
        """Takes a list of Measurement objects and merges the peaks dataframe of the given isotope."""
        df_list = []
        for m in measurements:
            df_list.append(m.peaks[isotope])
        df_merged = pd.concat(df_list, axis=0, ignore_index=True)
        return df_merged

    def reg_function(self, isotope: str, analyte_ppm_film: float = 0, force_zero: bool = False, mass_correction: float = 1):
        self.analyte_mass[isotope] = []
        self.analyte_intensity[isotope] = []

        if force_zero:
            self.analyte_mass[isotope].append(0)
            self.analyte_intensity[isotope].append(0)

        if type(self.__spotsizes[0]) == tuple:
            for i in range(len(self.__spotsizes)):
                x = self.__spotsizes[i][0]
                y = self.__spotsizes[i][1]
                z = self.__film_thickness[i]
                ablated_drymass = (z * 1e-7 * x * y * 1e-8) * (self.__film_concentration_percent / 100) * mass_correction
                if analyte_ppm_film != 0:
                    ablated_drymass = ablated_drymass * analyte_ppm_film * 1e-6
                self.analyte_mass[isotope].append(ablated_drymass)

        else:
            for i in range(len(self.__spotsizes)):
                d = self.__spotsizes[i]
                z = self.__film_thickness[i]
                ablated_drymass = (z * 1e-7 * (d/2)**2 * math.pi * 1e-8) * (self.__film_concentration_percent / 100) * mass_correction
                if analyte_ppm_film != 0:
                    ablated_drymass = ablated_drymass * analyte_ppm_film * 1e-6
                self.analyte_mass[isotope].append(ablated_drymass)

        for s in self.__standards:
            merged_df = self.merge_peak_df(measurements = list(s), isotope = isotope)
            self.analyte_intensity[isotope].append(merged_df['area'].mean())
            #self.analyte_intensity[isotope].append(s.peaks[isotope]['area'].mean())

        self.analyte_intensity[isotope] = [float(i) for i in self.analyte_intensity[isotope]]
        self.calibration[isotope] = sc.stats.linregress(x=self.analyte_mass[isotope], y=self.analyte_intensity[isotope])

        print(f'{isotope} calibration equation: y = {self.calibration[isotope].intercept:+.3e} {self.calibration[isotope].slope:+.3e} * x (R^2 = {(self.calibration[isotope].rvalue ** 2) * 100:.2f} %)')

        return self.calibration[isotope]


    def reg_plot_2(self, isotope: str):
        x = np.array(self.analyte_mass[isotope]) * 1e15
        y = self.analyte_intensity[isotope]
        plt.figure()
        plt.scatter(x, y, label='Data')
        # Fit line
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        # Calculate R^2
        y_pred = slope * np.array(x) + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        eqn_label = f'Fit: y = {intercept:.2e} + {slope:.2e}x\n$R^2$ = {r2:.3f}'
        plt.plot(x_fit, y_fit, color='red', label=eqn_label)
        plt.title(f"{isotope} calibration")
        plt.xlabel(f'mass (fg)')
        plt.ylabel('peak area (a.u.)')
        plt.xlim([0, max(x) * 1.1])
        plt.ylim([0, max(y) * 1.1])
        plt.legend()
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()

    def reg_plot(self, isotope: str, confidence_interval: int = 90):
        sns.regplot(x=self.analyte_mass[isotope], y=self.analyte_intensity[isotope], ci=confidence_interval)