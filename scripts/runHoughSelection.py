import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pythonHelpers.processing import run_hough_selection_data

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file',  type=str, required=True, help='Path to input file.')
    parser.add_argument('-parquet', '--output-parquet', type=str, required=True, help='Output Parquet filename')
    parser.add_argument('-o', '--output-root', type=str, help='Output ROOT filename for selected events')
    parser.add_argument('-s', '--start-event', type=int, help='Start event number.', default=0)
    parser.add_argument('-n', '--n-events', type=int, help='Number of events.', default=1000000)
    parser.add_argument('-f', '--fraction', type=float, default=1.0, help='Fraction of events to process')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_par_file = os.path.join(script_dir, '..', 'parFiles', 'TrackingParams.xml')

    parser.add_argument('-par', '--parFile', type=str, default=default_par_file, help='Tracking parameter file')
    parser.add_argument('-zmin', '--z-vtx-min', type=float, default=None, help='Minimum z-vertex intersection')
    parser.add_argument('-zmax', '--z-vtx-max', type=float, default=None, help='Maximum z-vertex intersection')
    parser.add_argument('-gal', '--gallery', type=str, help='Only process events listed in the provided gallery json file')
    return parser.parse_args()

def main():
    args = get_arguments()

    gal_file = None if args.gallery == "None" else args.gallery
    par_file = None if args.parFile == "None" else args.parFile

    run_hough_selection_data(
        input_file_path=args.input_file,
        output_parquet=args.output_parquet,
        output_root=args.output_root,
        start_event=args.start_event,
        n_events=args.n_events,
        fraction=args.fraction,
        par_file=par_file,
        z_vtx_min=args.z_vtx_min,
        z_vtx_max=args.z_vtx_max,
        gallery_file=gal_file
    )

if __name__ == "__main__":
    main()
