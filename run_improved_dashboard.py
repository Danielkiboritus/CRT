"""
Run the Improved CRT Trading Dashboard
"""

import argparse
from improved_dashboard import app

def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description="Run the Improved CRT Trading Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the dashboard on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Improved CRT Trading Dashboard")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {'Enabled' if args.debug else 'Disabled'}")
    print("=" * 50)
    print("Open your browser and navigate to:")
    print(f"http://{args.host}:{args.port}")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

if __name__ == "__main__":
    main()
