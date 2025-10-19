# src/asdm/simulator/app.py

import os
import sys
import socket
import tempfile
import webbrowser
import threading
import logging
import argparse
import uuid
import traceback
import multiprocessing
from flask import Flask, request, jsonify, make_response, render_template
from werkzeug.utils import secure_filename
from concurrent.futures import ProcessPoolExecutor

from asdm import sdmodel

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s - %(message)s')

executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

# A simple dictionary mapping a unique ID -> CSV content
DOWNLOAD_CACHE = {}

# Store pre-loaded model file path (if provided via CLI)
PRELOADED_MODEL_PATH = None

@app.route('/')
def index():
    return render_template("index.html", preloaded_model=PRELOADED_MODEL_PATH)

@app.route('/simulate', methods=['POST'])
def simulate_model():
    """
    Endpoint to handle the simulation request:
    - Receives an uploaded file
    - Uses a process pool to run the simulation
    - Returns the JSON result + a link to download CSV
    - Includes an error log section for debugging
    """
    if 'model_file' not in request.files:
        logging.error("No file part in request.")
        return jsonify({'error': 'No file found', 'error_log': 'No file uploaded'}), 400

    file = request.files['model_file']
    if file.filename == '':
        logging.error("Filename is empty.")
        return jsonify({'error': 'Empty filename', 'error_log': 'Uploaded file has no name'}), 400

    filename = secure_filename(file.filename)
    logging.info(f"Received file: {filename}")

    # Create a temporary directory to hold the file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        file.save(filepath)
        logging.info(f"File saved at: {filepath}")

        # Offload the simulation to the process pool
        future = executor.submit(run_simulation_and_csv, filepath)

        try:
            logging.debug("Starting simulation in a separate process...")
            df_records, csv_data, time_col = future.result()  # df_records is the JSON-friendly data, csv_data is the CSV, time_col is the time column name
            logging.debug("Simulation completed successfully.")
            error_log = ""  # No errors if successful
        except Exception as e:
            logging.exception("Error during simulation:")
            error_log = traceback.format_exc()  # Capture full traceback
            return jsonify({'error': str(e), 'error_log': error_log}), 500

        # Store the CSV data in memory with a unique ID
        download_id = str(uuid.uuid4())
        DOWNLOAD_CACHE[download_id] = csv_data

        return jsonify({
            "data": df_records,
            "time_col": time_col,
            "download_url": f"/download_csv/{download_id}",
            "error_log": error_log  # Include error logs even if empty
        })

def run_simulation_and_csv(filepath):
    """
    Runs simulation and also returns CSV data for download.
    """
    model = sdmodel(from_xmile=filepath)  # or from_xmile=filepath, whichever you use
    model.simulate()
    df = model.export_simulation_result(format='df')
    
    # Convert DataFrame to JSON-serialisable and CSV forms
    df_records = df.to_dict(orient='records')
    csv_data = df.to_csv(index=False)

    # Grabbing the time column name from sim_specs
    # (If it doesn't exist, fallback to "Time" or something else)
    time_col = model.sim_specs['time_units']

    return df_records, csv_data, time_col

@app.route('/download_csv/<download_id>')
def download_csv(download_id):
    """
    Serve the CSV file from memory when the user clicks "Download."
    """
    csv_data = DOWNLOAD_CACHE.get(download_id)
    if not csv_data:
        return "File not found or expired", 404
    
    # Optional: remove from cache to prevent indefinite memory usage
    # DOWNLOAD_CACHE.pop(download_id, None)
    
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=simulation_result.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/simulate_preloaded', methods=['POST'])
def simulate_preloaded():
    """
    Endpoint to simulate a pre-loaded model from CLI.
    Runs the model from its original location to preserve relative paths (e.g., CSV dependencies).
    """
    if not PRELOADED_MODEL_PATH:
        logging.error("No pre-loaded model available.")
        return jsonify({'error': 'No pre-loaded model', 'error_log': 'No model was pre-loaded via CLI'}), 404
    
    if not os.path.exists(PRELOADED_MODEL_PATH):
        logging.error(f"Pre-loaded model file not found: {PRELOADED_MODEL_PATH}")
        return jsonify({'error': 'Pre-loaded model file not found', 'error_log': f'File does not exist: {PRELOADED_MODEL_PATH}'}), 404
    
    logging.info(f"Simulating pre-loaded model: {PRELOADED_MODEL_PATH}")
    
    # Offload the simulation to the process pool
    future = executor.submit(run_simulation_from_path, PRELOADED_MODEL_PATH)
    
    try:
        logging.debug("Starting simulation in a separate process...")
        df_records, csv_data, time_col = future.result()
        logging.debug("Simulation completed successfully.")
        error_log = ""  # No errors if successful
    except Exception as e:
        logging.exception("Error during simulation:")
        error_log = traceback.format_exc()  # Capture full traceback
        return jsonify({'error': str(e), 'error_log': error_log}), 500
    
    # Store the CSV data in memory with a unique ID
    download_id = str(uuid.uuid4())
    DOWNLOAD_CACHE[download_id] = csv_data
    
    return jsonify({
        "data": df_records,
        "time_col": time_col,
        "download_url": f"/download_csv/{download_id}",
        "error_log": error_log  # Include error logs even if empty
    })

def run_simulation_from_path(filepath):
    """
    Runs simulation from the original file path.
    Changes working directory to the model's directory to resolve relative paths.
    """
    # Get the directory containing the model
    model_dir = os.path.dirname(os.path.abspath(filepath))
    original_cwd = os.getcwd()
    
    try:
        # Change to model directory so relative paths work
        os.chdir(model_dir)
        logging.info(f"Changed working directory to: {model_dir}")
        
        # Run simulation
        model = sdmodel(from_xmile=filepath)
        model.simulate()
        df = model.export_simulation_result(format='df')
        
        # Convert DataFrame to JSON-serialisable and CSV forms
        df_records = df.to_dict(orient='records')
        csv_data = df.to_csv(index=False)
        
        # Grabbing the time column name from sim_specs
        time_col = model.sim_specs['time_units']
        
        return df_records, csv_data, time_col
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)
        logging.info(f"Restored working directory to: {original_cwd}")

def open_browser(host, port):
    webbrowser.open_new(f"http://{host}:{port}")

def is_port_in_use(port):
    """Check if a port is in use (Cross-platform)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def run_simulator(host="127.0.0.1", port=8080, model_file=None):
    """
    Core function to run the ASDM simulator web server.
    
    Args:
        host (str): Host/IP address to bind to
        port (int): Port to run the server on
        model_file (str): Optional path to a model file to load automatically
    """
    global PRELOADED_MODEL_PATH
    
    # Validate and set pre-loaded model path if provided
    if model_file:
        abs_model_path = os.path.abspath(model_file)
        if not os.path.exists(abs_model_path):
            print(f"Error: Model file not found: {model_file}")
            sys.exit(1)
        
        if not (abs_model_path.endswith('.stmx') or abs_model_path.endswith('.xmile')):
            print(f"Error: Model file must be .stmx or .xmile format: {model_file}")
            sys.exit(1)
        
        PRELOADED_MODEL_PATH = abs_model_path
        logging.info(f"Pre-loading model: {abs_model_path}")
    
    # Check if the server is already running
    if is_port_in_use(port):
        print(f"ASDM simulator is already running on port {port}. Exiting.")
        return

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting ASDM simulator on {host}:{port} ...")

    threading.Timer(1, open_browser, [host, port]).start()

    app.run(debug=False, host=host, port=port)


def main():
    """Legacy entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Run the ASDM simulator web server.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host/IP address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to run the server on (default: 8080)")
    args = parser.parse_args()

    run_simulator(args.host, args.port)
