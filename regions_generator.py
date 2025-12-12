from shapely.geometry import box
from shapely.ops import unary_union
from PIL import Image, ImageDraw


def generate_regions_image(output_path='mumax_regions.png', geom_path='mumax_geometry.png',
                          num_insets=253, total_inset_distance=0.16, pixel_size=0.02, verbose=False):
    """
    Generate greyscale images for MuMax3: one for geometry, one for regions.
    
    Parameters
    ----------
    output_path : str
        Path to save the regions PNG image (grayscale 1-253)
    geom_path : str
        Path to save the geometry PNG image (0=device, 255=background)
    num_insets : int
        Number of inset regions to generate (default: 253)
    total_inset_distance : float
        Total distance (in microns) from edge to center over which to vary alpha (default: 0.15)
    pixel_size : float
        Size of each pixel in microns (default: 0.02)
    verbose : bool
        Print generation details (default: False)
    
    Returns
    -------
    tuple : (output_path, geom_path)
        Paths to the generated images
    """
    # All distances are in microns
    
    # Shape built from rectangles
    rectangles = [
        box(-1.5, -0.54, 0.5, 0.54),   # box(minx, miny, maxx, maxy) # Main arena 
        #box(-2.5, -13/60, -1.5, 13/60), # Input corridor 
        box(-2.5, -9/125, -1.5, 0.54), # Input corridor (shifted to top)
        box(0.5, -0.54, 2.5, -9/125), # Output corridor (bottom)
        box(0.5, 9/125, 2.5, 0.54), # Output corridor (top)
    ]
    
    shape = unary_union(rectangles)
    delta = total_inset_distance / num_insets  # inset distance per step
    
    inset_polygons = [shape]  # Start with the original shape as region 1
    current_shape = shape
    
    for i in range(num_insets - 1):  # num_insets - 1 because we already have the original shape
        # Negative buffer = inset
        inset = current_shape.buffer(-delta)
        if inset.is_empty:
            break  # nothing left to inset
        inset_polygons.append(inset)
        current_shape = inset
    
    # Define rasterization parameters
    margin = 10  # extra pixels around
    
    minx, miny, maxx, maxy = shape.bounds
    width  = int((maxx - minx) / pixel_size) + 2 * margin
    height = int((maxy - miny) / pixel_size) + 2 * margin
    
    # Create a white background image
    img = Image.new('L', (width, height), 255)  # 'L' = 8-bit grayscale, 255 = white
    draw = ImageDraw.Draw(img)
    
    def world_to_pixel(x, y):
        """Convert world coordinates to pixel coordinates (invert y for image coordinates)."""
        px = int((x - minx) / pixel_size) + margin
        py = int((maxy - y) / pixel_size) + margin  # note: y inverted
        return px, py
    
    # Draw from outermost inset to innermost
    n_layers = len(inset_polygons)
    for idx, poly in enumerate(inset_polygons):
        # Map layer index -> region ID (1 = outermost, 253 = innermost)
        # Background is 255, so avoid using 0 and 255 for regions
        region_id = idx + 1  # 1, 2, 3, ..., 253
    
        # Shapely polygons can be MultiPolygons; iterate over all parts
        if poly.geom_type == 'Polygon':
            polys = [poly]
        else:
            polys = list(poly)
    
        for p in polys:
            exterior_coords = [world_to_pixel(x, y) for x, y in p.exterior.coords]
            draw.polygon(exterior_coords, fill=region_id)
            for interior in p.interiors:
                interior_coords = [world_to_pixel(x, y) for x, y in interior.coords]
                draw.polygon(interior_coords, fill=255)  # holes as background
    
    # Create geometry image (0=device, 255=background)
    geom_img = Image.new('L', (width, height), 255)  # white background
    geom_draw = ImageDraw.Draw(geom_img)
    
    # Draw the entire device shape as black (0)
    if shape.geom_type == 'Polygon':
        polys = [shape]
    else:
        polys = list(shape)
    
    for p in polys:
        exterior_coords = [world_to_pixel(x, y) for x, y in p.exterior.coords]
        geom_draw.polygon(exterior_coords, fill=0)
        for interior in p.interiors:
            interior_coords = [world_to_pixel(x, y) for x, y in interior.coords]
            geom_draw.polygon(interior_coords, fill=255)  # holes as background
    
    if verbose:
        print(f"Generated {n_layers} regions (IDs 1-{n_layers})")
        print(f"Image size: {width}x{height} pixels")
        print(f"Pixel size: {pixel_size} µm/pixel")
    
    # Save both images
    img.save(output_path)
    geom_img.save(geom_path)
    
    return output_path, geom_path


if __name__ == '__main__':
    # Generate regions when run as a script
    generate_regions_image(verbose=True)
    print("Regions image saved to: mumax_regions.png")
