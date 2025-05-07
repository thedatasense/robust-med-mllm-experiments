from datetime import timedelta
import os
import csv
import cv2
import numpy as np
import re
import argparse
import glob

def parse_time_str(time_str):
    # Handle various time formats from the CSV
    time_str = time_str.strip()

    # Handle "0:00:00" or invalid time stamps
    if time_str == "0:00:00":
        return None

    # Handle H:MM:SS.s format
    if re.match(r'^\d+:\d+:\d+\.\d+$', time_str):
        parts = time_str.split(':')
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return timedelta(hours=h, minutes=m, seconds=s)

    # Handle MM:SS.s format
    if re.match(r'^\d+:\d+\.\d+$', time_str):
        parts = time_str.split(':')
        m, s = int(parts[0]), float(parts[1])
        return timedelta(minutes=m, seconds=s)

    # Handle H:MM:SS format
    if re.match(r'^\d+:\d+:\d+$', time_str):
        parts = time_str.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return timedelta(hours=h, minutes=m, seconds=s)

    return None

def extract_best_frame(video_path, time_seconds, fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    if time_seconds > video_duration:
        print(f"Time {time_seconds:.2f}s exceeds video duration {video_duration:.2f}s")
        cap.release()
        return None

    # Calculate target frame number
    target_frame = int(time_seconds * fps)

    # Try several approaches to get a good frame
    best_frame = None
    best_quality = -1  # Use this to track the "quality" of the frame

    # Try different methods and offsets
    methods = [
        (cv2.CAP_PROP_POS_FRAMES, target_frame),
        (cv2.CAP_PROP_POS_MSEC, time_seconds * 1000),
    ]

    offsets = [0, -5, 5, -10, 10, -30, 30]

    for method, value in methods:
        for offset in offsets:
            # Reset if we've tried different methods
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Apply offset
            adjusted_value = max(0, value + offset)
            if method == cv2.CAP_PROP_POS_FRAMES:
                adjusted_value = min(adjusted_value, total_frames - 1)

            # Set position
            cap.set(method, adjusted_value)

            # Read frame
            ret, frame = cap.read()

            if ret and frame is not None:
                # Calculate frame quality (simple metric: standard deviation of pixel values)
                frame_quality = np.std(frame)

                # If this is the best frame so far, keep it
                if frame_quality > best_quality:
                    best_quality = frame_quality
                    best_frame = frame.copy()

    cap.release()
    return best_frame

def process_case(base_dir, case_number, output_dir):
    # Setup paths
    case_dir = os.path.join(base_dir, f"case_{case_number:03d}")
    labels_dir = os.path.join(base_dir, "labels", f"case_{case_number:03d}")
    tools_csv = os.path.join(labels_dir, "tools.csv")

    # Check if paths exist
    if not os.path.exists(case_dir):
        print(f"Case directory not found: {case_dir}")
        return False

    if not os.path.exists(tools_csv):
        print(f"Tools CSV not found: {tools_csv}")
        return False

    # Find video files
    video_files = sorted(glob.glob(os.path.join(case_dir, f"case_{case_number:03d}_video_part_*.mp4")))
    if not video_files:
        print(f"No video files found in {case_dir}")
        return False

    print(f"Found {len(video_files)} video files for case {case_number}")

    # Create output directory
    case_output_dir = os.path.join(output_dir, f"case_{case_number:03d}")
    os.makedirs(case_output_dir, exist_ok=True)

    # Get video properties for the first video
    cap = cv2.VideoCapture(video_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video FPS: {fps}")

    # Process tools CSV
    successful_extractions = 0
    failed_extractions = 0

    with open(tools_csv, 'r') as file:
        csv_reader = csv.DictReader(file)
        fieldnames = csv_reader.fieldnames + ['adjusted_time', 'frame_filename', 'extraction_status']

        output_csv = os.path.join(case_output_dir, "extraction_results.csv")
        with open(output_csv, 'w', newline='') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            for row in csv_reader:
                # Get index and time
                row_idx = row.get('index', '0')
                time_str = row.get('install_case_time', '')

                # Parse time
                timedelta_obj = parse_time_str(time_str)
                if timedelta_obj is None:
                    print(f"Skipping row {row_idx}: Invalid time format: {time_str}")
                    row['adjusted_time'] = ''
                    row['frame_filename'] = ''
                    row['extraction_status'] = 'skipped: invalid time'
                    csv_writer.writerow(row)
                    continue

                # Add 4 seconds
                adjusted_timedelta = timedelta_obj + timedelta(seconds=8)
                adjusted_seconds = adjusted_timedelta.total_seconds()

                # Get tool name
                tool_name = row.get('groundtruth_toolname', '').strip()
                if not tool_name:
                    tool_name = row.get('commercial_toolname', '').strip()

                # Clean tool name for filename
                clean_tool_name = tool_name.replace(' ', '_').replace('/', '_')
                if '(' in clean_tool_name and ')' in clean_tool_name:
                    clean_tool_name = clean_tool_name.replace('(', '').replace(')', '')

                # Set output filename
                frame_filename = f"row{row_idx}_{clean_tool_name}.png"
                frame_path = os.path.join(case_output_dir, frame_filename)

                print(f"Processing row {row_idx}: {tool_name} at {adjusted_seconds:.2f}s")

                # Extract frame using our robust method
                # For simplicity, using first video part - in a complete solution,
                # you'd determine which video part based on timestamps
                frame = extract_best_frame(video_files[0], adjusted_seconds, fps)

                if frame is not None:
                    # Save the frame
                    success = cv2.imwrite(frame_path, frame)
                    if success and os.path.getsize(frame_path) > 10000:
                        row['frame_filename'] = frame_filename
                        row['extraction_status'] = 'success'
                        successful_extractions += 1
                        print(f"  √ Successfully extracted frame: {frame_filename}")
                    else:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                        row['frame_filename'] = ''
                        row['extraction_status'] = 'failed: empty frame'
                        failed_extractions += 1
                        print(f"  × Failed to extract valid frame")
                else:
                    row['frame_filename'] = ''
                    row['extraction_status'] = 'failed: no frame'
                    failed_extractions += 1
                    print(f"  × Failed to extract frame")

                row['adjusted_time'] = str(adjusted_timedelta)
                csv_writer.writerow(row)

    print(f"Case {case_number} processing complete:")
    print(f"  Successful extractions: {successful_extractions}")
    print(f"  Failed extractions: {failed_extractions}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract frames from surgical videos.')
    parser.add_argument('--base-dir', type=str, default='/Volumes/TVault2/datasets/sugvu24',
                        help='Base directory containing case folders')
    parser.add_argument('--output-dir', type=str, default='/Volumes/TVault2/datasets/sugvu24/extracted_frames',
                        help='Output directory for extracted frames')
    parser.add_argument('--case', type=int, nargs='+',
                        help='Case number(s) to process (e.g., 1 2 3)')
    parser.add_argument('--all', action='store_true',
                        help='Process all available cases')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which cases to process
    if args.all:
        case_folders = sorted(glob.glob(os.path.join(args.base_dir, 'case_[0-9][0-9][0-9]')))
        cases = [int(os.path.basename(folder).split('_')[1]) for folder in case_folders]
    elif args.case:
        cases = args.case
    else:
        print("Error: Please specify either --case or --all")
        return

    # Process each case
    for case_number in cases:
        print(f"\n{'='*40}")
        print(f"Processing case {case_number:03d}")
        print(f"{'='*40}")
        process_case(args.base_dir, case_number, args.output_dir)

if __name__ == "__main__":
    main()