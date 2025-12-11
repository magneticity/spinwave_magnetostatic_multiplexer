def build_mumax_script(params, dot_positions):
    """
    Build MuMax3 script string from parameters and dot positions.
    params: dict with keys:
      nx, ny, nz, dx, dy, dz,
      geom_png, regions_ovf,
            Msat_device, Aex_device, edge_alpha, device_alpha,
            Msat_dot, Aex_dot, alpha_dot,
      alpha_k, alpha_width_m,  # width in meters used for normalized mapping if needed
      stripline_x, stripline_width, stripline_height,
            f1, f2, T, sample_dt,
            Bz_ext, By_amp,
      n_dots, dot_diameter
    dot_positions: list[(x,y)] in meters
    """
    nx = params.get('nx', 250)
    ny = params.get('ny', 50)
    nz = params.get('nz', 20)
    dx = params.get('dx', 20e-9)
    dy = params.get('dy', 20e-9)
    dz = params.get('dz', 10e-9)
    geom_png = params.get('geom_png', 'mumax_geometry.png')
    regions_ovf = params.get('regions_ovf', 'regions_map.ovf')
    Msat_device = params.get('Msat_device', 1.4e5)
    Aex_device = params.get('Aex_device', 3.5e-12)
    edge_alpha = params.get('edge_alpha', 0.5)
    device_alpha = params.get('device_alpha', 2e-4)
    Msat_dot = params.get('Msat_dot', 1.145e6)
    Aex_dot = params.get('Aex_dot', 7.5e-12)
    alpha_dot = params.get('alpha_dot', 0.2)
    alpha_k = params.get('alpha_k', -5.0)
    stripline_x = params.get('stripline_x', -2.5e-6 + 150e-9)
    stripline_y = params.get('stripline_y', 0)
    stripline_width = params.get('stripline_width', 300e-9)
    stripline_height = params.get('stripline_height', 0.8e-6)
    f1 = params.get('f1', 2.6e9)
    f2 = params.get('f2', 2.8e9)
    T = params.get('T', 50e-9)
    sample_dt = params.get('sample_dt', 50e-12)
    Bz_ext = params.get('Bz_ext', 0.2)
    By_amp = params.get('By_amp', 0.1e-3)
    dot_diam = params.get('dot_diameter', 100e-9)

    # Build dot additions
    dot_lines = []
    for (x,y) in dot_positions:
        dot_lines.append(f"dots = dots.add((cylinder({dot_diam},{dot_diam}).transl({x:.15e}, {y:.15e}, 0)).transl(0, 0, 50e-9))")
    dots_block = "\n    ".join(dot_lines) if dot_lines else ""

    script = f"""
// Mesh & sizes
SetGridsize({nx}, {ny}, {nz})
SetCellsize({dx}, {dy}, {dz})

// Geometry & regions
device_geom := (ImageShape(\"{geom_png}\")).sub(cuboid({nx}*{dx}, {ny}*{dy}, {nz}*{dz}/2).transl(0, 0, {nz}*{dz}/4))
regions.LoadFile(\"{regions_ovf}\")

// Material params
Msat = {Msat_device}
Aex  = {Aex_device}
edge_alpha := {edge_alpha}
device_alpha := {device_alpha}
alpha = edge_alpha
k := {alpha_k}
for i:=1; i<254; i+=1{{
    s := i/253
    new_alpha := edge_alpha + (device_alpha - edge_alpha) * ((exp(k*s) - 1) / (exp(k) - 1))
    alpha.setRegion(i, new_alpha)
}}

// Dots region 254
dots := cylinder(0, 0)
if {len(dot_positions)} != 0 {{
    {dots_block}
    defregion(254, dots)
    Msat.setRegion(254, {Msat_dot})
    Aex.setRegion(254, {Aex_dot})
    alpha.setRegion(254, {alpha_dot})
}}

// Input stripline region 255
input_stripline := cuboid({stripline_width}, {stripline_height}, {dz* nz}).transl({stripline_x}, {stripline_y}, {-dz* nz/2})
defregion(255,input_stripline)

setgeom(device_geom.add(dots))
saveas(geom, "geom")
saveas(regions, "regions_map")
saveas(alpha, "alpha_map")

// Simulation
T = {T}
f1 := {f1}
f2 := {f2}
sample_dt := {sample_dt}
m = uniform(0.02, 0.02, 1)
B_ext = vector(0, 0, {Bz_ext})
autosave(m,sample_dt)
TableAutosave(sample_dt)
B_ext.setregion(255, vector(0, ({By_amp})*sin(2*pi*f1*t) + ({By_amp})*sin(2*pi*f2*t), {Bz_ext}))

run(T)
"""
    return script
