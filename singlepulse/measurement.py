from . import utils
import pandas as pd
import numpy as np
import scipy as sc
import plotly.graph_objects as go


class SyngistixMeasurement:
    """
    The basic concept evolves around two DataFrames self.data and self.peaks which are filled with data when applying methods.
    Methods return the specific calculations in addition to saving the data to these DataFrames.
    """

    def __init__(self, file_path: str):
        """
        Initialize a regular Syngistix file or Nanomodul file (set nanomodul == True).
        """
        self.file_name = file_path.name
        self.__timescale_count = 0
        self.__nanomodul = utils.check_nanomodul(file_path)
        if self.__nanomodul:
            self.data = pd.read_csv(file_path, skiprows=0).iloc[:, :-1]
            self.measured_isotopes = self.data.columns
        else:
            self.data = pd.read_csv(file_path, skiprows=2)
            self.measured_isotopes = self.data.loc[:, self.data.columns != "Time in Seconds "].columns
        self.peaks = dict()
        for isotope in self.measured_isotopes:
            self.peaks[isotope] = pd.DataFrame(columns=['index', 'time', 'height', 'width', 'background', 'area'])

    def timescale(self, isotope: str, cycle_time: float = False):
        """
        Add a timescale for a specific isotope to the dataframe.
        Enter a cycle_time if nanomodul files are used.
        """
        assert isotope in self.measured_isotopes, f"Isotope '{isotope}' was not measured! {self.file_name}"
        if self.__nanomodul:
            assert cycle_time, "Please specify cycle time when using Nanomodul files!"
            self.__cycle_time = cycle_time
            timestamps = [0 + self.__timescale_count * self.__cycle_time / len(self.measured_isotopes)]
            for i in range(len(self.data) - 1):
                timestamps.append(timestamps[i] + self.__cycle_time)
            self.data[f"{isotope}_time"] = timestamps
            self.__timescale_count += 1
            return True
        else:
            self.data["Time in Seconds "] = np.linspace(0, max(self.data["Time in Seconds "]), len(self.data))
            self.__cycle_time = self.data["Time in Seconds "][1] - self.data["Time in Seconds "][0]
            timestamps = [0 + self.__timescale_count * self.__cycle_time / len(self.measured_isotopes)]
            for i in range(len(self.data) - 1):
                timestamps.append(timestamps[i] + self.__cycle_time)
            self.data[f"{isotope}_time"] = timestamps
            self.__timescale_count += 1
            return True

    def savgol(self, isotope: str, window_length: int = 100, polyorder: int = 2, deriv: int = 0):
        """
        Smooth the data using the scipy.savgol_filter() method. Check method parameters if desired.
        """
        self.data[f"{isotope}_savgol"] = sc.signal.savgol_filter(self.data[isotope], window_length=window_length, polyorder=polyorder, deriv=deriv)
        return True

    def global_background(self, isotope: str, start: float = 0.1, end: float = 1):
        start_idx = int(start // self.__cycle_time)
        end_idx = int(end // self.__cycle_time)
        self.__global_bg = np.median(self.data[isotope][start_idx:end_idx])
        self.data[f"{isotope}_corr"] = self.data[f"{isotope}"] - self.__global_bg
        return True

    def peak_finding(self, isotope: str, threshold: float = 50, distance: float = 10e-3, width: float = 10e-6, savgol: bool = False):
        """
        Find peaks using scipy.signal.find_peaks(). Savitzky-Golay signal can be used optionally by setting savgol == True. Check find_peaks() settings if desired.
        """
        assert f'{isotope}_time' in self.data.columns, f"Calculate '{isotope}_time' before peak detection!"
        if savgol:
            self.savgol()
            signal = self.data[f"{isotope}_savgol"]
        else:
            signal = self.data[f"{isotope}"]
            self.peaks[isotope]['index'] = sc.signal.find_peaks(signal,
                                                                height=threshold,
                                                                distance=distance/self.__cycle_time,
                                                                width=width/self.__cycle_time)[0]

        peak_idx = self.peaks[isotope]['index']
        self.peaks[isotope]['time'] = self.data[f'{isotope}_time'].iloc[peak_idx].tolist()
        self.peaks[isotope]['height'] = self.data[isotope].iloc[peak_idx].tolist()
        return True
    
    def peak_width(self, isotope: str, criterion: int = 10):
        """
        Calculate peak width using scipy.signal.peak_widths() method.
        Default value for criterion is 10 for 10-percent criterion.
        """
        width = sc.signal.peak_widths(x=self.data[f'{isotope}'], peaks=self.peaks[isotope][f'index'], rel_height=(1 - criterion / 100))

        peakwidth = width[0] * self.__cycle_time
        left = [i*self.__cycle_time for i in width[2]]
        right = [i*self.__cycle_time for i in width[3]]

        assert len(left) == len(right), f"Left and right peak width arrays have different lengths! {self.file_name}"
        self.peaks[isotope]['width'] = peakwidth
        self.peaks[isotope]['time_left'] = left
        self.peaks[isotope]['time_right'] = right

        return True

    def peak_background(self, isotope: str, distance: float = 0, window_size: float = 3):
        """
        Calculates the peak background left and right to the peak.
        The distance_from_peak specifies how far the background window is from the peak boundaries.
        The window_size specifies how many peak widths the background window is wide (on each side).
        """
        backgrounds = []
        self.__bgwindowsize = window_size
        self.__bgfactor = distance

        for i in range(len(self.peaks[isotope]['index'])):
            peak_width = self.peaks[isotope]['width'][i]
            peak_time = self.peaks[isotope]['time'][i]
            peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
            asymmetry = (peak_time_right - peak_time) / peak_width

            bgbound_left_outer_idx = (peak_time - peak_width * (self.__bgfactor * (1 - asymmetry) + self.__bgwindowsize)) / self.__cycle_time
            bgbound_left_inner_idx = (peak_time - peak_width * self.__bgfactor * (1 - asymmetry)) / self.__cycle_time
            left_data = self.data[isotope][int(bgbound_left_outer_idx):int(bgbound_left_inner_idx)]

            bgbound_right_inner_idx = (peak_time + peak_width * self.__bgfactor * asymmetry) / self.__cycle_time
            bgbound_right_outer_idx = (peak_time + peak_width * (self.__bgfactor * asymmetry + self.__bgwindowsize)) / self.__cycle_time
            right_data = self.data[isotope][int(bgbound_right_inner_idx):int(bgbound_right_outer_idx)]

            backgrounds.append((left_data.mean(), right_data.mean()))

        self.peaks[isotope]['background'] = [i+j for i, j in backgrounds]

        return True

    def peak_area(self, isotope: str, mode: str = 'trapezoid', resize: float = 1, local_background: bool = False):
        """
        Calculates peak area using scipy 'trapezoid' or 'romberg'.
        If local_background = True, the peak_background() has to be calculated beforehand.
        If local_background is not specified or False, the global background is used (calculated from second 0.1 to 1).
        The integration window is peak_width * resize_factor.
        """
        self.__intfactor = resize
        areas = []
        for i in range(len(self.peaks[isotope]['index'])):
            peak_time = self.peaks[isotope]['time'][i]
            peak_width = self.peaks[isotope]['width'][i]
            peak_background = self.peaks[isotope]['background'][i]
            peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
            asymmetry = (peak_time_right - peak_time) / peak_width

            intbound_left_idx = int((peak_time - peak_width * self.__intfactor * (1 - asymmetry)) / self.__cycle_time)
            intbound_right_idx = int((peak_time + peak_width * self.__intfactor * asymmetry) / self.__cycle_time)

            if local_background:
                #assert peak_time.empty is False and not peak_background.isna().all(), f"Calculate local background before integration or set local_background = False!"
                signal = [k - self.peaks[isotope]['background'][i] for k in self.data[isotope][intbound_left_idx:intbound_right_idx]]
            else:
                signal = [k - self.global_background(isotope, start=0.1, end=1) for k in self.data[isotope][intbound_left_idx:intbound_right_idx]]

            if mode == 'trapezoid':
                areas.append(sc.integrate.trapezoid(signal))
            elif mode == 'romberg':
                areas.append(sc.integrate.romberg(signal))

        self.peaks[isotope]['area'] = areas
        return True

    def calibrate(self, isotope, calibration: object):
        """
        Uses a Calibration object to append 'mass' column to the peaks dataframe.
        """
        self.peaks[isotope]['mass'] = (self.peaks[isotope]['area'] - calibration.calibration[isotope].intercept) / calibration.calibration[isotope].slope
        return True


    def area_ratio(self, isotope_one: str, gravfac_one: float, isotope_two: str, gravfac_two: float):
        """
        Calculate the area ratio between two isotopes - for every peak separately. 
        This is achieved by looking at every peak of the large area isotope and checking if there was an adjacent smaller area peak found. 
        If yes, the ratio is calculated and stored in self.__ratios. 
        :param isotope_one: The isotope with the supposed larger area.
        :param isotope_two: The isotope with the smaller area.
        :return: Ratio between the two areas as float.
        """
        ratios = []
        for i_one in range(len(self.peaks[isotope_one])):
            area_one = self.peaks[isotope_one]["area"][i_one]

            left_timestamp = self.peaks[isotope_one]["time_left"][i_one]
            right_timestamp = self.peaks[isotope_one]["time_right"][i_one]

            signal = self.data[(left_timestamp < self.data["Time in Seconds "]) & (self.data["Time in Seconds "] < right_timestamp)][isotope_two]

            area_two = sc.integrate.trapezoid(signal)

            ratio = area_one * gravfac_one / (area_one * gravfac_one + area_two * gravfac_two)
            if ratio != 1:
                ratios.append(ratio)

        ratio_mean = np.mean(ratios)
        ratio_stdev = np.std(ratios)

        return (ratios, ratio_mean, ratio_stdev)
    
    def plot(self, isotope: str, fig: object, savgol: bool = False, integration: bool = False, peaks: bool = False, background: bool = False, width: bool = False):
        """
        Plot the data and optional elements such as integration boundaries or local background.
        """

        fig.update_layout(height=900, xaxis_title="Time (s)", yaxis_title="Intensity")
        fig.update_traces(line_width=1, selector=dict(type='scatter'), showlegend=True)

        traces = []
        shapes = []

        assert f'{isotope}_time' in self.data.columns, f"'{isotope}_time' required!"
        original_trace = go.Scatter(line=dict(width=1.5), x=self.data[f"{isotope}_time"], y=self.data[f"{isotope}"], mode='lines', name=f"{isotope} ({self.file_name})")
        traces.append(original_trace)

        if savgol:
            assert f'{isotope}_savgol' in self.data.columns, f"Use savgol() before plotting Savitzky-Golay signal!"
            savgol_trace = go.Scatter(line=dict(width=1.5), x=self.data[f"{isotope}_time"], y=self.data[f"{isotope}_savgol"], mode='lines', name=f"'{self.file_name}' ({isotope}_savgol)")
            traces.append(savgol_trace)

        try:
            peak_index = self.peaks[isotope]['index']
            peak_time = self.peaks[isotope]['time']
            peak_height = self.peaks[isotope]['height']
            peak_width = self.peaks[isotope]['width']
            peak_background = self.peaks[isotope]['background']
        except:
            pass

        if peaks:
            peak_trace = go.Scatter(x=peak_time, y=peak_height, mode='markers', marker_color="red", name=f"{isotope} peaks ({self.file_name})")
            traces.append(peak_trace)

        for i in range(len(peak_index)):
            if width:
                assert peak_time.empty is False and not peak_width.isna().all(), "Use peak_finding() and peak_width() before plotting peak widths!"
                peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                width_shape = dict(type='rect', x0=peak_time_left, x1=peak_time_right, y0=0, y1=peak_height[i], line_width=0, fillcolor='blue', opacity=0.1)
                shapes.append(width_shape)

            if background:
                assert peak_time.empty is False and not peak_background.isna().all(), "Use peak_finding() and peak_background() before plotting backgrounds!"
                bgleft, bgright = self.__backgrounds[i]
                peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                bgbound_left_outer_time = peak_time[i] - peak_width[i] * (self.__bgfactor * (1 - asymmetry) + self.__bgwindowsize)
                bgbound_left_inner_time = peak_time[i] - peak_width[i] * self.__bgfactor * (1 - asymmetry)
                bgbound_right_inner_time = peak_time[i] + peak_width[i] * self.__bgfactor * asymmetry
                bgbound_right_outer_time = peak_time[i] + peak_width[i] * (self.__bgfactor * asymmetry + self.__bgwindowsize)
                bg_shape_left = dict(type="line", y0=bgleft, y1=bgleft, x0=bgbound_left_outer_time, x1=bgbound_left_inner_time, line=dict(color="red", width=3))
                bg_shape_right = dict(type="line", y0=bgright, y1=bgright, x0=bgbound_right_inner_time, x1=bgbound_right_outer_time, line=dict(color="red", width=3))
                shapes.append(bg_shape_left)
                shapes.append(bg_shape_right)

            if integration:
                assert peak_time.empty is False and not peak_width.isna().all(), "Use peak_finding() and peak_width() before plotting integration boundaries!"
                peak_time_left, peak_time_right = self.peaks[isotope]['time_left'][i], self.peaks[isotope]['time_right'][i]
                asymmetry = (peak_time_right - peak_time[i]) / peak_width[i]
                intbound_left_time = peak_time[i] - peak_width[i] * (1 - asymmetry) * self.__intfactor
                intbound_right_time = peak_time[i] + peak_width[i] * asymmetry * self.__intfactor
                width_shape = dict(type='rect', x0=intbound_left_time, x1=intbound_right_time, y0=0, y1=peak_height[i], line_width=0,
                                   fillcolor='green', opacity=0.1)
                shapes.append(width_shape)

        for trace in traces:
            fig.add_trace(trace)
        for shape in shapes:
            fig.add_shape(shape)