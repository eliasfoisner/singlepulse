import pandas as pd
import numpy as np


class GDCParticles:
    def __init__(self, measurement: object, ce_isotope: str = "CeO/156", gd_isotope: str = "GdO/174", time_window: float = 0.1):
        self.__measurement = measurement
        self.__ce_isotope = ce_isotope
        self.__gd_isotope = gd_isotope

        ce_df = measurement.peaks[ce_isotope][["time", "area"]].copy()
        ce_df["isotope"] = ce_isotope
        gd_df = measurement.peaks[gd_isotope][["time", "area"]].copy()
        gd_df["isotope"] = gd_isotope

        ce_df = ce_df.reset_index(drop=True)
        gd_df = gd_df.reset_index(drop=True)

        # we need to ensure that only entries are copied where both Ce and Gd are found:
        pairs = []
        for _, ce_row in ce_df.iterrows():
            gd_candidates = gd_df[np.abs(gd_df["time"] - ce_row["time"]) <= time_window]
            
            if not gd_candidates.empty:
                gd_row = gd_candidates.iloc[(np.abs(gd_candidates["time"] - ce_row["time"])).argmin()]
                if not pd.isna(ce_row["time"]) and not pd.isna(gd_row["time"]):
                    pairs.append({
                        "ce_time": ce_row["time"],
                        "ce_area": ce_row["area"],
                        "gd_time": gd_row["time"],
                        "gd_area": gd_row["area"]
                    })
        pairs = [p for p in pairs if all(pd.notna([p["ce_time"], p["gd_time"]]))]
        self.particles = pd.DataFrame(pairs)

    def constituent_mass(self, calibration: object):
        """
        Uses a Calibration object to append 'mass' column to the particles dataframe.
        """
        y_ce = self.particles['ce_area']
        d_ce = calibration.calibration[self.__ce_isotope].intercept
        k_ce = calibration.calibration[self.__ce_isotope].slope
        self.particles['ce_mass'] = (y_ce - d_ce) / k_ce

        y_gd = self.particles['gd_area']
        d_gd = calibration.calibration[self.__gd_isotope].intercept
        k_gd = calibration.calibration[self.__gd_isotope].slope
        self.particles['gd_mass'] = (y_gd - d_gd) / k_gd
        return True
    
    def particle_mass(self, mw_ce, mw_gd):
        """
        Computes the mass of the GDC particles from the constituent masses.
        In this calculation, oxygen is also considered.
        """
        mw_o = 16.0
        m_ce = self.particles['ce_mass']
        m_gd = self.particles['gd_mass']
        n_ce = m_ce / mw_ce
        n_gd = m_gd / mw_gd
        x = n_gd / (n_ce + n_gd)
        n_ce_formula = 1 - x
        n_gd_formula = x
        n_o_formula = 2 * (n_ce_formula + n_gd_formula) - 0.5 * x
        o_ratio = n_o_formula / (n_ce_formula + n_gd_formula)
        n_o = (n_ce + n_gd) * o_ratio
        m_o = n_o * mw_o
        self.particles['mass'] = m_ce + m_gd + m_o
        return True

    def particle_diameter(self, density: float = 7.32):
        """
        Computes the diameter of the GDC particles from the mass and density
        :param density: Density of the GDC particles in g/cm^3
        :return: Diameter in nm
        """
        rho = density
        m = self.particles['mass']
        d = (6 * m / (rho * np.pi)) ** (1/3) * 1e7
        self.particles['diameter'] = d
        return True