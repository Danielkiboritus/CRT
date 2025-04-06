"""
🚀 CRT Trading System

This script runs the CRT trading system, which can operate in two modes:
1. Standalone: CRT agent runs independently and executes trades
2. Integrated: CRT agent provides signals to the trading agent

The system uses the same risk management framework regardless of mode.
"""

import os
import time
from datetime import datetime, timedelta
from termcolor import colored, cprint
import logging
import argparse

# Local imports
from src.agents.crt_agent import CRTAgent
from src.agents.trading_agent import TradingAgent
from src.agents.risk_agent import RiskAgent
from src.integrations.crt_integration import CRTIntegration
from src.config.crt_config import RUN_CRT_STANDALONE, CRT_PROVIDE_SIGNALS, CRT_CHECK_INTERVAL_MINUTES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crt_system.log')
    ]
)
logger = logging.getLogger(__name__)

def run_standalone_mode():
    """Run CRT agent in standalone mode"""
    logger.info("Starting CRT agent in standalone mode")
    
    # Initialize agents
    crt_agent = CRTAgent()
    risk_agent = RiskAgent()
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cprint(f"\n⏰ CRT System Run Starting at {current_time}", "white", "on_green")
            
            # Check risk limits first
            cprint("\n🛡️ Checking risk limits...", "white", "on_blue")
            risk_breach = risk_agent.run()
            
            if risk_breach:
                cprint("\n⚠️ Risk limits breached! Skipping CRT agent run.", "white", "on_red")
                time.sleep(CRT_CHECK_INTERVAL_MINUTES * 60)
                continue
            
            # Run CRT agent
            cprint("\n🕯️ Running CRT agent...", "white", "on_blue")
            crt_agent.run()
            
            # Calculate next run time
            next_run = datetime.now() + timedelta(minutes=CRT_CHECK_INTERVAL_MINUTES)
            cprint(f"\n⏳ CRT System run complete. Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "white", "on_green")
            
            # Sleep until next interval
            time.sleep(CRT_CHECK_INTERVAL_MINUTES * 60)
        
        except KeyboardInterrupt:
            cprint("\n👋 CRT System shutting down gracefully...", "white", "on_blue")
            break
        
        except Exception as e:
            logger.error(f"Error in standalone mode: {str(e)}")
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")
            time.sleep(300)  # Sleep for 5 minutes on error

def run_integrated_mode():
    """Run CRT agent integrated with trading agent"""
    logger.info("Starting CRT agent in integrated mode")
    
    # Initialize agents
    crt_integration = CRTIntegration()
    trading_agent = TradingAgent()
    risk_agent = RiskAgent()
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cprint(f"\n⏰ CRT System Run Starting at {current_time}", "white", "on_green")
            
            # Check risk limits first
            cprint("\n🛡️ Checking risk limits...", "white", "on_blue")
            risk_breach = risk_agent.run()
            
            if risk_breach:
                cprint("\n⚠️ Risk limits breached! Skipping trading cycle.", "white", "on_red")
                time.sleep(CRT_CHECK_INTERVAL_MINUTES * 60)
                continue
            
            # Get CRT signals
            cprint("\n🕯️ Getting CRT signals...", "white", "on_blue")
            crt_signals = crt_integration.get_crt_signals()
            
            # Format signals for trading agent
            strategy_signals = crt_integration.format_for_trading_agent(crt_signals)
            
            # Run trading agent with CRT signals
            cprint("\n🤖 Running trading agent with CRT signals...", "white", "on_blue")
            trading_agent.run_trading_cycle(strategy_signals)
            
            # Calculate next run time
            next_run = datetime.now() + timedelta(minutes=CRT_CHECK_INTERVAL_MINUTES)
            cprint(f"\n⏳ CRT System run complete. Next run at {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "white", "on_green")
            
            # Sleep until next interval
            time.sleep(CRT_CHECK_INTERVAL_MINUTES * 60)
        
        except KeyboardInterrupt:
            cprint("\n👋 CRT System shutting down gracefully...", "white", "on_blue")
            break
        
        except Exception as e:
            logger.error(f"Error in integrated mode: {str(e)}")
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")
            time.sleep(300)  # Sleep for 5 minutes on error

def main():
    """Main function to run the CRT system"""
    parser = argparse.ArgumentParser(description='Run the CRT Trading System')
    parser.add_argument('--mode', type=str, choices=['standalone', 'integrated', 'auto'], 
                        default='auto', help='Mode to run the system in')
    
    args = parser.parse_args()
    
    # Print banner
    cprint("\n" + "="*50, "cyan")
    cprint("🌙 Moon Dev's CRT Trading System 🕯️", "white", "on_blue")
    cprint("="*50 + "\n", "cyan")
    
    # Determine mode
    mode = args.mode
    if mode == 'auto':
        # Use configuration settings to determine mode
        if RUN_CRT_STANDALONE:
            mode = 'standalone'
        elif CRT_PROVIDE_SIGNALS:
            mode = 'integrated'
        else:
            cprint("⚠️ Both standalone and integrated modes are disabled in config!", "white", "on_red")
            cprint("🔧 Defaulting to standalone mode", "white", "on_yellow")
            mode = 'standalone'
    
    # Run in selected mode
    if mode == 'standalone':
        cprint("🚀 Running in STANDALONE mode", "white", "on_green")
        run_standalone_mode()
    else:
        cprint("🚀 Running in INTEGRATED mode", "white", "on_green")
        run_integrated_mode()

if __name__ == "__main__":
    main()
