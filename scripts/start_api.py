#!/usr/bin/env python3
import argparse
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Risk Scoring API server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    load_dotenv()
    
    import uvicorn
    uvicorn.run(
        "packages.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()