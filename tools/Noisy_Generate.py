"""
Python implementation of the RRest synthetic ECG and PPG generators.

Examples:

  Generate a clean ECG signal of 4 seconds, 80 bpm heart rate, 500 sampling rate and amplitude 1.

    ecg_t, ecg_v = gen_ecg_signal(4, 80, 500, 1)

  To generate a PPG signal instead:

    ppg_t, ppg_v = gen_ppg_signal(4, 80, 500, 1)

  To add baseline noise via modulation:

    bw_t, bw_v = gen_bw_noise(ecg_t, ecg_v, fs=0.8)

  To add amplitude modulation noise:

    am_t, am_v = gen_am_noise(ecg_t, ecg_v, fs=0.8)

  To add frequency modulation noise:

    fm_t, fm_v = gen_fm_noise(ecg_t, ecg_v, fs=0.8, amplitude=0.5)

  All three types of noise can be added at the same time:

    noisy_t, noisy_v = gen_bw_noise(ppg_t, ppg_v, fs=0.8)
    noisy_t, noisy_v = gen_am_noise(noisy_t, noisy_v, fs=0.8)
    noisy_t, noisy_v = gen_fm_noise(noisy_t, noisy_v, fs=0.4, amplitude=0.06)

Sources:
    http://peterhcharlton.github.io/RRest/synthetic_dataset.html

References:
    http://dx.doi.org/10.1007/s10877-011-9332-y
    http://dx.doi.org/10.1007/978-3-319-18191-2_10

Author:
    Horacio Sanson (Allm Inc.)
"""

import numpy as np
from scipy.signal import detrend

def gen_ppg_pulse():
    """
    Generates a single clean PPG pulse.
    """
    ppg_t = np.arange(0, 1.01, 0.01, dtype=np.float32)
    ppg_v = np.array([1077,1095.27676998271,1150.39696200268,1253.64011384634,1410.01430649103,1614.23738042021,
        1852.88862847947,2107.19963850395,2354.10827203910,2571.83504395768,2747.58603743220,2878.78047565119,
        2967.12445239160,3018.75510162723,3042.94721680823,3047.45001341082,3037.62026345592,3018.14128598677,
        2991.80738548651,2959.68011205818,2924.12507822977,2885.14270668177,2843.86782834153,2802.08903767062,
        2759.59973773619,2716.75011175682,2673.89981522322,2630.12661808970,2584.21488466353,2533.75439619717,
        2480.71246051141,2423.10076799285,2360.92007271860,2294.87110683952,2226.11520653275,2155.41275815825,
        2085.84032425344,2019.82560572195,1959.08788519998,1905.61966740180,1860.15018328118,1823.59858526554,
        1799.16523439130,1785.68124422590,1781.40496155451,1785.65000894081,1795.60758813888,1810.23123770638,
        1825.88000834451,1843.33369762174,1863.68739383102,1877.70370775466,1882.41001847768,1880.67124988824,
        1874.06744888836,1862.89381761287,1847.14991953269,1829.89768909817,1810.34993145378,1788.72101296379,
        1765.94996721702,1742.03613261809,1717.39751028193,1692.75877484727,1668.68005245276,1644.48098438338,
        1619.84239547012,1595.97728854980,1572.41003606020,1548.84233116767,1525.27517210550,1501.70741230815,
        1478.27021279132,1455.77367020325,1433.54980777827,1412.12496274662,1390.69993145582,1369.27500149014,
        1348.40755237670,1328.05378822197,1308.39978542052,1289.88870123678,1272.51984025750,1256.29379973178,
        1241.20988734577,1229.73332273703,1217.97235166145,1206.26252309710,1195.54992698555,1185.27003397509,
        1176.98743219884,1169.12989658769,1160.98994575908,1153.99245954403,1146.99245931931,1138.20624497094,
        1127.13495738213,1112.99225070036,1097.13740724184,1083.85480956071,1080])
    return [ppg_t, ppg_v]

def gen_ecg_pulse():
    """
    Generates a single clean ECG pulse.
    """
    ecg_t = np.arange(0, 1.01, 0.01, dtype=np.float32)
    ecg_v = np.array([
        -0.120000000000000,-0.120000000000000,-0.125000000000000,-0.122500113257034,-0.125000000000000,
        -0.120000000000000,-0.125000000000000,-0.117500065569862,-0.125000000000000,-0.117499958278698,
        -0.120000000000000,-0.120000000000000,-0.124999856955537,-0.122500005960186,-0.105000131124091,
        -0.115000000000000,-0.110000000000000,-0.122499946358326,-0.125000000000000,-0.109999558946239,
        -0.100000357611158,-0.0899996065808297,-0.0750000000000000,-0.0775001728453928,-0.0850000000000000,
        -0.0975001490224131,-0.100000000000000,-0.0925001251639052,-0.0949997377518178,-0.0824998986647591,
        -0.0800000000000000,-0.0800000000000000,-0.100000429133389,-0.0299993562231754,0.464997210632973,
         0.997500745023246,0.954992990821313,0.134999821173104,-0.194999845035165,-0.165000035761116,
        -0.119999880796281,-0.117499958273724,-0.110000202646322,-0.107499934437955,-0.109999928477768,
        -0.100000000000000,-0.100000000000000,-0.0900000000000000,-0.0949999761592562,-0.0824998390749790,
        -0.0800000000000000,-0.0675001609250210,-0.0649999761592562,-0.0574998867564668,-0.0500000000000000,
        -0.0400000000000000,-0.0300000000000000,-0.0124999344379545,0.00500020264632248,0.0275000417262755,
         0.0649997615925615,0.100000035761115,0.145000309929670,0.189999988078207,0.224999666229586,
         0.252499970199070,0.270000000000000,0.275000000000000,0.250000214566695,0.197500232447252,
         0.114999761592563,0.0325003040057223,-0.0299992132554528,-0.0849997496721898,-0.105000023840744,
        -0.130000000000000,-0.130000000000000,-0.140000000000000,-0.130000000000000,-0.127500196709585,
        -0.120000000000000,-0.120000000000000,-0.110000000000000,-0.110000000000000,-0.105000000000000,
        -0.105000000000000,-0.100000000000000,-0.100000000000000,-0.100000000000000,-0.100000000000000,
        -0.104999821194421,-0.110000000000000,-0.110000000000000,-0.105000000000000,-0.114999773512934,
        -0.120000000000000,-0.115000000000000,-0.122500113257034,-0.115000000000000,-0.120000000000000,
        -0.110000000000000], dtype=np.float32)
    return([ecg_t, ecg_v])

def normalize_pulse(pulse_v, amplitude = 1):
    range = np.max(pulse_v) - np.min(pulse_v)
    return amplitude * (pulse_v - min(pulse_v))/range

def resample_pulse(pulse_t, pulse_v, hr = 80, fs = 500):
    """
    Resample data at appropriate sampling rate
    """
    beat_duration = 60/hr
    beat_samples = beat_duration * fs
    interval = (1/(beat_samples - 1))
    t_new = np.arange(0, 1+interval, interval, dtype=np.float32)
    resampled_pulse = np.interp(t_new, pulse_t, pulse_v)
    return resampled_pulse, t_new

def gen_signal(pulse_t, pulse_v, secs = 210, hr = 80, fs = 500, amplitude = 1):
    """
    Generates a signal by repeating the input pulse. If the input pulse is an
    ECG pulse the generated signal is an ECG signal. Likewise if the input pulse
    is a PPG pulse, then the generated signal is a PPG signal.

    Params:
      pulse_t: Time axis of the pulse.
      pulse_v: Value axis of the pulse.
      secs: Lenght in seconds of the generated signal.
      hr: Heart rate of the generated signal.
      fs: Sampling rate of the generated signal.
      amplitude: Amplitude of the generated signal.
    """
    sig_v, sig_t = resample_pulse(pulse_t, normalize_pulse(pulse_v, amplitude), hr, fs)
    num_samples = secs * fs
    num_reps = np.ceil(num_samples / len(sig_t))
    sig_v = np.tile(sig_v, (1, int(num_reps))).flatten()
    sig_v = sig_v[0:num_samples]
    sig_t = np.arange(0,(num_samples))*(1/len(sig_t))*(len(sig_t)/fs)
    return [sig_t, sig_v]

def gen_ecg_signal(secs = 210, hr = 80, fs = 500, amplitude = 1):
    """
    Generates a synthetic ECG signal

    Params:
      secs: Length in seconds of the generated signal.
      hr: Heart rate in bpm of the generated signal.
      fs: Sampling frequency of the generated signal.
      amplitude: Amplitude of the generated signal.
    """
    pulse_t, pulse_v = gen_ecg_pulse()
    return gen_signal(pulse_t, pulse_v, secs, hr, fs, amplitude)

def gen_ppg_signal(secs = 210, hr = 80, fs = 500, amplitude = 1):
    """
    Generates a synthetic PPG signal

    Params:
      secs: Length in seconds of the generated signal.
      hr: Heart rate in bpm of the generated signal.
      fs: Sampling frequency of the generated signal.
      amplitude: Amplitude of the generated signal.
    """
    pulse_t, pulse_v = gen_ppg_pulse()
    return gen_signal(pulse_t, pulse_v, secs, hr, fs, amplitude)

def gen_bw_noise(sig_t, sig_v, fs, amplitude=0.1):
    """
    Adds baseline wandering to the input signal.

    Parameters:
      fs: Wandering frequency in Hz
      amplitude: Wandering amplitude
    """
    w = 2*np.pi*fs
    mod_v = sig_v + amplitude * np.sin(w*sig_t)
    return [sig_t, mod_v]

def gen_am_noise(sig_t, sig_v, fs, amplitude=0.1):
    """
    Adds amplitude modulation noise to the input signal.

    Parameters:
      fs: Modulation frequency in Hz
      amplitude: Modulation amplitude
    """
    w = 2*np.pi*fs
    mod_v = detrend(sig_v)*((1/amplitude)+np.cos(w*sig_t));
    return [sig_t, mod_v]

def gen_fm_noise(sig_t, sig_v, fs, amplitude=0.05):
    """
    Adds frequency modulation noise to the input signal.

    Parameters:
      fs: Modulation frequency in Hz
      amplitude: Modulation amplitude
    """
    w = 2*np.pi*fs
    mod_t = sig_t + amplitude*np.sin(w*sig_t)
    mod_v = np.interp(mod_t, sig_t, sig_v)
    return [sig_t, mod_v]

if __name__ == "__main__":
    import tempfile
    import matplotlib.pyplot as plt

    ecg_t, ecg_v = gen_ecg_signal()

    bw_t, bw_v = gen_bw_noise(ecg_t, ecg_v, fs=0.8)
    am_t, am_v = gen_am_noise(ecg_t, ecg_v, fs=0.8)
    fm_t, fm_v = gen_fm_noise(ecg_t, ecg_v, fs=0.8, amplitude=0.08)

    noisy_t, noisy_v = gen_bw_noise(ecg_t, ecg_v, fs=0.8)
    noisy_t, noisy_v = gen_am_noise(noisy_t, noisy_v, fs=0.8)
    noisy_t, noisy_v = gen_fm_noise(noisy_t, noisy_v, fs=0.4, amplitude=0.06)

    # plt.clf()
    # plt.cla()
    f, axarr = plt.subplots(5, sharex=True)
    plt.figure(figsize=(30, 12), dpi=100)
    axarr[0].plot(ecg_t[0:5000], ecg_v[0:5000])
    axarr[0].set_title("Clean ECG signal")
    axarr[1].plot(bw_t[0:5000], bw_v[0:5000])
    axarr[1].set_title("Baseline wander noise")
    axarr[2].plot(am_t[0:5000], am_v[0:5000])
    axarr[2].set_title("Amplitude modulation noise")
    axarr[3].plot(fm_t[0:5000], fm_v[0:5000])
    axarr[3].set_title("Frequency modulation noise")
    axarr[4].plot(noisy_t[0:5000], noisy_v[0:5000])
    axarr[4].set_title("Noisy ECG signal")

    ecg_file = tempfile.gettempdir() + "/ecg_sig.png"
    print("Generating ECG plot at " + ecg_file)
    #plt.savefig(ecg_file)

    plt.show()

    # ppg_t, ppg_v = gen_ppg_signal()
    #
    # bw_t, bw_v = gen_bw_noise(ppg_t, ppg_v, fs=0.8)
    # am_t, am_v = gen_am_noise(ppg_t, ppg_v, fs=0.8)
    # fm_t, fm_v = gen_fm_noise(ppg_t, ppg_v, fs=0.8, amplitude=0.08)
    #
    # noisy_t, noisy_v = gen_bw_noise(ppg_t, ppg_v, fs=0.8)
    # noisy_t, noisy_v = gen_am_noise(noisy_t, noisy_v, fs=0.8)
    # noisy_t, noisy_v = gen_fm_noise(noisy_t, noisy_v, fs=0.4, amplitude=0.06)
    #
    # # plt.clf()
    # # plt.cla()
    # f, axarr = plt.subplots(5, sharex=True)
    # axarr[0].plot(ppg_t[0:5000], ppg_v[0:5000])
    # axarr[0].set_title("Clean PPG signal")
    # axarr[1].plot(bw_t[0:5000], bw_v[0:5000])
    # axarr[1].set_title("Baseline wander noise")
    # axarr[2].plot(am_t[0:5000], am_v[0:5000])
    # axarr[2].set_title("Amplitude modulation noise")
    # axarr[3].plot(fm_t[0:5000], fm_v[0:5000])
    # axarr[3].set_title("Frequency modulation noise")
    # axarr[4].plot(noisy_t[0:5000], noisy_v[0:5000])
    # axarr[4].set_title("Noisy PPG signal")
    #
    # ppg_file = tempfile.gettempdir() + "/ppg_sig.png"
    # print("Generating PPG plot at " + ppg_file)
    # #plt.savefig(ppg_file)
    # plt.show()
    # print()