"""
Worker module for parallel simulation evaluation.
Contains all functions needed by worker processes.
"""

import numpy as np
from scipy.signal.windows import hann, hamming
from subprocess import run, PIPE, STDOUT
from glob import glob
from os import path
from mumax_script import build_mumax_script


def read_mumax3_table(filename):
    """Puts the mumax3 output table in a pandas dataframe"""
    from pandas import read_table
    table = read_table(filename)
    table.columns = ' '.join(table.columns).split()[1::2]
    return table
    

def read_mumax3_ovffiles(outputdir):
    """Load all ovffiles in outputdir into a dictionary of numpy arrays"""
    # convert all ovf files in the output directory to numpy files
    p = run(["./mumax3-convert","-numpy",outputdir+"/*.ovf"], stdout=PIPE, stderr=STDOUT)
    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))
    # read the numpy files (the converted ovf files)
    fields = {}
    for npyfile in glob(outputdir+"/*.npy"):
        key = path.splitext(path.basename(npyfile))[0]
        fields[key] = np.load(npyfile)
    return fields


def build_magnetization_array(fields, pattern='m'):
    """Build a 5D magnetization array from MuMax3 output fields."""
    import re
    
    # Find all matching fields and extract timestamps
    time_fields = {}
    for key, data in fields.items():
        match = re.match(rf'{pattern}(\d{{6}})$', key)
        if match:
            timestamp = int(match.group(1))
            time_fields[timestamp] = data
    
    if not time_fields:
        raise ValueError(f"No fields matching pattern '{pattern}' found")
    
    # Sort by timestamp
    timestamps = sorted(time_fields.keys())
    
    # Get dimensions from first field
    first_field = time_fields[timestamps[0]]
    ncomp, nz, ny, nx = first_field.shape
    nt = len(timestamps)
    
    if ncomp != 3:
        raise ValueError(f"Expected 3 components, got {ncomp}")
    
    # Build the 5D array
    M = np.zeros((ncomp, nz, ny, nx, nt))
    for idx, t in enumerate(timestamps):
        M[:, :, :, :, idx] = time_fields[t]
    
    return M, timestamps


def mag_tfft_select(M, component, dt, fsel, dimorder='zyx', detrend=True, 
                    window='hann', stat='amp'):
    """Extract amplitude/power maps at specified frequencies."""
    # Parse component
    if isinstance(component, str):
        comp_map = {'x': 0, 'y': 1, 'z': 2}
        ci = comp_map.get(component.lower())
        if ci is None:
            raise ValueError("component must be 'x', 'y', 'z' or 0, 1, 2")
    elif component in [0, 1, 2]:
        ci = component
    else:
        raise ValueError("component must be 'x', 'y', 'z' or 0, 1, 2")
    
    # Extract component: M is (comp, z, y, x, t)
    ncomp, nz, ny, nx, nt = M.shape
    X = M[ci, :, :, :, :]  # (z, y, x, t)
    
    # Reorder spatial dimensions according to dimorder
    dimorder_map = {
        'xyz': [2, 1, 0, 3],
        'xzy': [2, 0, 1, 3],
        'yxz': [1, 2, 0, 3],
        'yzx': [1, 0, 2, 3],
        'zxy': [0, 2, 1, 3],
        'zyx': [0, 1, 2, 3],
    }
    
    if dimorder.lower() not in dimorder_map:
        raise ValueError(f"Invalid dimorder: {dimorder}")
    
    perm = dimorder_map[dimorder.lower()]
    X = np.transpose(X, perm)
    
    nd1, nd2, nd3, nt = X.shape
    nvox = nd1 * nd2 * nd3
    
    # Reshape to (nt, nvox) for efficient FFT along time axis
    X = X.reshape(nd1 * nd2 * nd3, nt).T
    
    # Detrend: remove mean
    if detrend:
        X = X - np.nanmean(X, axis=0, keepdims=True)
    
    # Apply window
    if window.lower() == 'hann':
        w = hann(nt, sym=False)
    elif window.lower() == 'hamming':
        w = hamming(nt, sym=False)
    else:
        w = np.ones(nt)
    
    X = X * w[:, np.newaxis]
    
    # FFT along time axis
    F = np.fft.fft(X, axis=0)
    
    # Frequency bins
    fs = 1.0 / dt
    fbin = np.fft.fftfreq(nt, dt)
    
    # Find nearest bins for requested frequencies
    fsel = np.atleast_1d(fsel)
    k = np.round(fsel / (fs / nt)).astype(int)
    k = np.clip(k, 0, nt - 1)
    fpos = k * (fs / nt)
    
    # Extract frequency slices and reshape to 3D spatial maps
    maps = []
    for ki in k:
        row = F[ki, :]
        vol = row.reshape(nd1, nd2, nd3)
        
        if stat.lower() == 'complex':
            maps.append(vol)
        elif stat.lower() == 'amp':
            maps.append(np.abs(vol))
        elif stat.lower() == 'power':
            maps.append(np.abs(vol)**2)
        else:
            raise ValueError("stat must be 'complex', 'amp', or 'power'")
    
    return maps, fpos


def run_mumax3(script, name, verbose=True):
    """Executes a mumax3 script and convert ovf files to numpy files"""
    scriptfile = name + ".txt"
    outputdir  = name + ".out"
    
    # write the input script in scriptfile
    with open(scriptfile, 'w') as f:
        f.write(script)
    
    # call mumax3 to execute this script
    p = run(["./mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
    if verbose or p.returncode != 0:
        print(p.stdout.decode('UTF-8'))
    
    if path.exists(outputdir + "/table.txt"):
        table = read_mumax3_table(outputdir + "/table.txt")
    else:
        table = None
    
    fields = read_mumax3_ovffiles(outputdir)
    return table, fields


def measure_region_amplitude(map_3d, center_x, center_y, size, device_size_x, device_size_y):
    """Measure the total amplitude in a rectangular region of a 3D map."""
    nz, ny, nx = map_3d.shape
    
    # Convert physical coordinates to pixel indices
    x_min_phys = center_x - size/2
    x_max_phys = center_x + size/2
    y_min_phys = center_y - size/2
    y_max_phys = center_y + size/2
    
    # Convert to pixel indices
    ix_min = int((x_min_phys + device_size_x/2) / device_size_x * nx)
    ix_max = int((x_max_phys + device_size_x/2) / device_size_x * nx)
    iy_min = int((y_min_phys + device_size_y/2) / device_size_y * ny)
    iy_max = int((y_max_phys + device_size_y/2) / device_size_y * ny)
    
    # Clip to valid range
    ix_min = max(0, min(ix_min, nx-1))
    ix_max = max(0, min(ix_max, nx-1))
    iy_min = max(0, min(iy_min, ny-1))
    iy_max = max(0, min(iy_max, ny-1))
    
    # Extract region and sum over all dimensions
    region = map_3d[:, iy_min:iy_max+1, ix_min:ix_max+1]
    total_amplitude = np.sum(np.abs(region))
    
    return total_amplitude, (ix_min, ix_max, iy_min, iy_max)


def evaluate_individual(dot_positions, sim_name, params):
    """
    Evaluate a single individual (set of dot positions).
    This function runs the simulation and calculates fitness.
    
    Parameters
    ----------
    dot_positions : list
        List of (x, y) tuples for dot positions
    sim_name : str
        Name for this simulation
    
    Returns
    -------
    result : dict
        Dictionary with 'fitness' and other metrics
    """
    # Ensure n_dots matches provided positions
    params = dict(params)
    params['n_dots'] = len(dot_positions)
    script = build_mumax_script(params, dot_positions)
    
    # Run simulation
    table, fields = run_mumax3(script, name=sim_name, verbose=False)
    
    # Build magnetization array and perform FFT
    M, timestamps = build_magnetization_array(fields, pattern='m')
    maps, fpos = mag_tfft_select(M, component='y', dt=params.get('sample_dt', 50e-12), fsel=[params.get('f1', 2.6e9), params.get('f2', 2.8e9)], 
                                 dimorder='zyx', detrend=True, window='hann', stat='amp')
    
    # Measure outputs
    measurement_size = params.get('detector_size', 300e-9)
    device_size_x = params['dx'] * params['nx']
    device_size_y = params['dy'] * params['ny']
    right_edge_x = device_size_x / 2
    x_shift = params.get('detector_offset_x_cells', 0) * params['dx']
    y_shift_top = params.get('detector_top_offset_y_cells', 0) * params['dy']
    y_shift_bottom = params.get('detector_bottom_offset_y_cells', 0) * params['dy']
    output_top_center = (right_edge_x - 1.0e-6 + x_shift, 0.35e-6 + y_shift_top)
    output_bottom_center = (right_edge_x - 1.0e-6 + x_shift, -0.35e-6 + y_shift_bottom)
    
    results = {}
    for i, (fmap, f) in enumerate(zip(maps, fpos)):
        freq_label = f"{f/1e9:.2f} GHz"
        
        amp_top, _ = measure_region_amplitude(
            fmap, output_top_center[0], output_top_center[1], 
            measurement_size, device_size_x, device_size_y
        )
        
        amp_bottom, _ = measure_region_amplitude(
            fmap, output_bottom_center[0], output_bottom_center[1],
            measurement_size, device_size_x, device_size_y
        )
        
        results[freq_label] = {'top': amp_top, 'bottom': amp_bottom}
    
    # Calculate fitness combining selectivity and total output magnitude
    f1_label = f"{fpos[0]/1e9:.2f} GHz"
    f2_label = f"{fpos[1]/1e9:.2f} GHz"
    
    # Selectivity: how well each frequency routes to its intended output
    selectivity_top = results[f1_label]['top'] / (results[f2_label]['top'] + 1e-10)
    selectivity_bottom = results[f2_label]['bottom'] / (results[f1_label]['bottom'] + 1e-10)
    selectivity_score = selectivity_top * selectivity_bottom
    
    # Total output magnitude: sum of correctly routed signals
    # We want f1 (2.6 GHz) to go to top and f2 (2.8 GHz) to go to bottom
    total_output = results[f1_label]['top'] + results[f2_label]['bottom']
    
    # Normalize total output by a reference value to keep it in reasonable range
    # Use 1e-3 as approximate expected order of magnitude
    output_magnitude_score = total_output / 1e-3
    
    # Combined fitness: product of selectivity and output magnitude
    # This encourages both good routing AND strong output signals
    fitness = selectivity_score * output_magnitude_score
    
    return {
        'fitness': fitness,
        'selectivity_score': selectivity_score,
        'output_magnitude_score': output_magnitude_score,
        'total_output': total_output,
        'selectivity_top': selectivity_top,
        'selectivity_bottom': selectivity_bottom,
        'results': results
    }
