# src/asdm/cli.py

"""
ASDM Command Line Interface

Main entry point for the unified ASDM CLI tool.
"""

import sys
import argparse
from asdm import __version__


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='asdm',
        description='ASDM - A Python package for System Dynamics Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  asdm simulator                    Launch the web-based simulator
  asdm simulator model.stmx         Launch simulator and load model.stmx
  asdm simulator --port 9000        Launch simulator on custom port
  asdm run model.stmx               Simulate model.stmx and save results as model.csv
  asdm run model.stmx --output out  Simulate and save results as out.csv
  asdm --version                    Show version information

For more information: https://github.com/wzh1895/ASDM
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'ASDM v{__version__}'
    )
    
    # Create subparsers for subcommands
    subparsers = parser.add_subparsers(
        title='Available commands',
        dest='command',
        help='Command to execute',
        metavar='<command>'
    )
    
    # Add simulator subcommand
    simulator_parser = subparsers.add_parser(
        'simulator',
        help='Launch the ASDM web-based simulator',
        description='Start a local web server with an interactive System Dynamics model simulator.'
    )
    
    simulator_parser.add_argument(
        'model_file',
        nargs='?',
        default=None,
        help='Optional: Path to model file (.stmx or .xmile) to load and run automatically'
    )
    
    simulator_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host/IP address to bind to (default: 127.0.0.1)'
    )
    
    simulator_parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to run the server on (default: 8080)'
    )
    
    # Add run subcommand
    run_parser = subparsers.add_parser(
        'run',
        help='Run a model simulation from command line',
        description='Load a System Dynamics model, simulate it with default settings, and export results to CSV.'
    )
    
    run_parser.add_argument(
        'model_file',
        help='Path to model file (.stmx or .xmile) to simulate'
    )
    
    run_parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output CSV file name (default: same as model file with .csv extension)'
    )
    
    return parser


def cmd_simulator(args):
    """Handle the 'simulator' subcommand."""
    from asdm.simulator.app import run_simulator
    run_simulator(args.host, args.port, args.model_file)


def cmd_run(args):
    """Handle the 'run' subcommand."""
    from pathlib import Path
    from asdm import sdmodel
    
    # Get model file path
    model_path = Path(args.model_file)
    
    # Check if model file exists
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Determine output file name
    if args.output:
        output_path = Path(args.output)
        # Add .csv extension if not present
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')
        elif output_path.suffix != '.csv':
            output_path = output_path.with_suffix(output_path.suffix + '.csv')
    else:
        # Use model file name with .csv extension
        output_path = model_path.with_suffix('.csv')
    
    print(f"Loading model: {model_path}")
    
    try:
        # Load the model
        model = sdmodel(from_xmile=model_path)
        print("Model loaded successfully!")
        
        # Simulate the model
        print("Running simulation...")
        model.simulate()
        
        # Export results
        print("Exporting results...")
        result = model.export_simulation_result(format='df')
        
        # Save to CSV
        result.to_csv(output_path)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        sys.exit(1)


def main():
    """Main entry point for the unified ASDM CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate command handler
    if args.command == 'simulator':
        cmd_simulator(args)
    elif args.command == 'run':
        cmd_run(args)
    else:
        # This shouldn't happen given the subparsers, but just in case
        parser.print_help()
        sys.exit(1)


def main_legacy():
    """
    Legacy entry point for 'asdm.simulator' command.
    Shows deprecation warning and redirects to new command.
    """
    import warnings
    
    # Show deprecation warning
    print("=" * 70)
    print("⚠️  DEPRECATION WARNING")
    print("=" * 70)
    print("The command 'asdm.simulator' is deprecated and will be removed soon.")
    print("Please use the new command instead:")
    print()
    print("  New:  asdm simulator")
    print("  Old:  asdm.simulator  (this command)")
    print()
    print("All functionality remains the same, just the command name changes.")
    print("=" * 70)
    print()
    
    # Run the simulator with the legacy entry point
    from asdm.simulator.app import main as legacy_main
    legacy_main()


if __name__ == '__main__':
    main()

