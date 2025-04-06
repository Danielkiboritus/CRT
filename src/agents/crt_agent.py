"""
🕯️ Moon Dev's CRT Trading Agent

This agent implements the Candle Range Theory (CRT) trading strategy
with AI-enhanced decision making and risk management.

It can run as a standalone agent or provide signals to other agents.
"""

import anthropic
import os
import pandas as pd
import json
from termcolor import colored, cprint
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import logging

# Local imports
from src.config import *
from src import nice_funcs as n
from src.data.ohlcv_collector import collect_all_tokens
from src.strategies.crt_strategy import CRTStrategy
from src.agents.base_agent import BaseAgent

# Load environment variables
load_dotenv()

# CRT Agent Prompt
CRT_ANALYSIS_PROMPT = """
You are Moon Dev's CRT Pattern Analysis AI 🕯️

Analyze the provided Candle Range Theory (CRT) pattern and market data to validate the pattern quality and trading opportunity.

CRT Pattern Details:
{pattern_details}

Market Data:
{market_data}

Consider the following factors:
1. Pattern quality and clarity
2. Proximity to key levels
3. Market structure alignment
4. Volume confirmation
5. Risk-reward ratio
6. Recent price action and momentum
7. Potential for false breakout

Respond in this exact format:

1. First line must be one of: VALID, QUESTIONABLE, or INVALID (in caps)

2. Then explain your reasoning, including:
- Pattern quality assessment
- Key level analysis
- Market structure alignment
- Volume analysis
- Risk factors
- Confidence level (as a percentage, e.g. 75%)

Remember:
- Moon Dev always prioritizes risk management! 🛡️
- High-quality CRT patterns occur at key levels
- Volume should confirm the reversal
- Market structure should align with the pattern direction
- Risk-reward ratio should be at least 2:1
"""

class CRTAgent(BaseAgent):
    def __init__(self):
        """Initialize Moon Dev's CRT Agent"""
        super().__init__('crt')  # Initialize base agent with type
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_KEY")
        if not api_key:
            raise ValueError("🚨 ANTHROPIC_KEY not found in environment variables!")
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Initialize CRT strategy
        self.strategy = CRTStrategy()
        
        # Initialize signals storage
        self.signals = []
        self.validated_signals = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        cprint("🕯️ CRT Agent initialized!", "white", "on_blue")
    
    def analyze_pattern(self, pattern, market_data):
        """Use AI to analyze and validate a CRT pattern"""
        try:
            # Format pattern details for AI
            pattern_details = json.dumps(pattern, indent=2)
            
            # Format market data for AI
            market_data_str = json.dumps(market_data, indent=2)
            
            # Create the prompt
            prompt = CRT_ANALYSIS_PROMPT.format(
                pattern_details=pattern_details,
                market_data=market_data_str
            )
            
            # Get AI analysis
            message = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                temperature=AI_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response
            response = message.content
            if isinstance(response, list):
                # Extract text from TextBlock objects if present
                response = '\n'.join([
                    item.text if hasattr(item, 'text') else str(item)
                    for item in response
                ])
            
            lines = response.split('\n')
            validation = lines[0].strip() if lines else "INVALID"
            
            # Extract confidence from the response
            confidence = 0
            for line in lines:
                if 'confidence' in line.lower():
                    # Extract number from string like "Confidence: 75%"
                    try:
                        confidence = int(''.join(filter(str.isdigit, line)))
                    except:
                        confidence = 50  # Default if not found
            
            # Add validation results to pattern
            pattern['ai_validation'] = validation
            pattern['ai_confidence'] = confidence
            pattern['ai_reasoning'] = '\n'.join(lines[1:]) if len(lines) > 1 else "No detailed reasoning provided"
            
            self.logger.info(f"AI Pattern Analysis: {validation} with {confidence}% confidence")
            
            return pattern
        
        except Exception as e:
            self.logger.error(f"Error in AI pattern analysis: {str(e)}")
            pattern['ai_validation'] = "ERROR"
            pattern['ai_confidence'] = 0
            pattern['ai_reasoning'] = f"Error during analysis: {str(e)}"
            return pattern
    
    def get_market_data(self, symbol):
        """Get market data for a symbol"""
        try:
            # Get data for higher timeframe (4h)
            df_high = n.get_data(symbol, 2, '4h')  # 2 days of 4h data
            
            # Get data for lower timeframe (1h)
            df_low = n.get_data(symbol, 1, '1h')  # 1 day of 1h data
            
            return {
                'high_timeframe': df_high,
                'low_timeframe': df_low
            }
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def generate_signals(self, symbol):
        """Generate CRT signals for a symbol"""
        try:
            # Get market data
            market_data = self.get_market_data(symbol)
            if not market_data:
                self.logger.warning(f"No market data available for {symbol}")
                return []
            
            # Run CRT strategy
            signals = self.strategy.run(
                market_data['high_timeframe'],
                market_data['low_timeframe']
            )
            
            self.logger.info(f"Generated {len(signals)} CRT signals for {symbol}")
            return signals
        
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
    
    def validate_signals(self, signals, symbol):
        """Validate signals using AI"""
        validated_signals = []
        
        for signal in signals:
            try:
                # Get additional market data for context
                market_data = self.get_market_data(symbol)
                
                # Analyze pattern with AI
                validated_signal = self.analyze_pattern(signal, market_data)
                
                # Only include VALID signals
                if validated_signal['ai_validation'] == 'VALID':
                    validated_signals.append(validated_signal)
            
            except Exception as e:
                self.logger.error(f"Error validating signal: {str(e)}")
        
        self.logger.info(f"Validated {len(validated_signals)} out of {len(signals)} signals")
        return validated_signals
    
    def execute_signal(self, signal, symbol):
        """Execute a trading signal"""
        try:
            # Skip if in backtest mode
            if BACKTEST_MODE:
                self.logger.info(f"[BACKTEST] Would execute {signal['action']} for {symbol} at {signal['entry']}")
                return True
            
            # Execute the trade
            if signal['action'] == 'BUY':
                # Calculate position size based on risk
                risk_amount = self.calculate_risk_amount()
                stop_distance = abs(signal['entry'] - signal['stop_loss'])
                position_size = risk_amount / stop_distance
                
                # Execute buy
                self.logger.info(f"Executing BUY for {symbol} at {signal['entry']}")
                # Implement actual trade execution here
                
            elif signal['action'] == 'SELL':
                # Calculate position size based on risk
                risk_amount = self.calculate_risk_amount()
                stop_distance = abs(signal['entry'] - signal['stop_loss'])
                position_size = risk_amount / stop_distance
                
                # Execute sell
                self.logger.info(f"Executing SELL for {symbol} at {signal['entry']}")
                # Implement actual trade execution here
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error executing signal: {str(e)}")
            return False
    
    def calculate_risk_amount(self):
        """Calculate risk amount based on account balance"""
        try:
            # Get account balance
            balance = n.get_portfolio_value()
            
            # Calculate risk amount (1% of balance by default)
            risk_amount = balance * (RISK_PERCENT / 100)
            
            return risk_amount
        
        except Exception as e:
            self.logger.error(f"Error calculating risk amount: {str(e)}")
            return 100  # Default risk amount
    
    def format_signals_for_trading_agent(self):
        """Format CRT signals for the trading agent"""
        formatted_signals = {}
        
        for signal in self.validated_signals:
            symbol = signal.get('symbol', 'UNKNOWN')
            
            if symbol not in formatted_signals:
                formatted_signals[symbol] = []
            
            formatted_signals[symbol].append({
                'strategy': 'CRT',
                'signal_type': signal['type'],
                'action': signal['action'],
                'entry': signal['entry'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'confidence': signal['ai_confidence'],
                'risk_reward': signal['risk_reward'],
                'time': signal['time']
            })
        
        return formatted_signals
    
    def run(self):
        """Run the CRT agent"""
        try:
            self.logger.info("🕯️ Running CRT Agent...")
            
            # Clear previous signals
            self.signals = []
            self.validated_signals = []
            
            # Process each monitored token
            for symbol in MONITORED_TOKENS:
                # Skip excluded tokens
                if symbol in EXCLUDED_TOKENS:
                    continue
                
                self.logger.info(f"Processing {symbol}...")
                
                # Generate signals
                symbol_signals = self.generate_signals(symbol)
                
                # Add symbol to signals
                for signal in symbol_signals:
                    signal['symbol'] = symbol
                
                # Add to all signals
                self.signals.extend(symbol_signals)
                
                # Validate signals with AI
                validated = self.validate_signals(symbol_signals, symbol)
                self.validated_signals.extend(validated)
                
                # Execute signals if running as standalone
                if RUN_CRT_STANDALONE:
                    for signal in validated:
                        self.execute_signal(signal, symbol)
            
            # Format signals for trading agent
            formatted_signals = self.format_signals_for_trading_agent()
            
            self.logger.info(f"CRT Agent run complete. Found {len(self.signals)} signals, {len(self.validated_signals)} validated.")
            
            return formatted_signals
        
        except Exception as e:
            self.logger.error(f"Error running CRT agent: {str(e)}")
            return {}

def main():
    """Main function to run the CRT agent"""
    cprint("🕯️ CRT Agent Starting...", "white", "on_blue")
    
    agent = CRTAgent()
    
    while True:
        try:
            # Run the agent
            signals = agent.run()
            
            # Print results
            cprint(f"\n📊 CRT Agent Results:", "white", "on_green")
            for symbol, symbol_signals in signals.items():
                cprint(f"\n{symbol}: {len(symbol_signals)} signals", "white", "on_blue")
                for signal in symbol_signals:
                    cprint(f"  {signal['action']} at {signal['entry']:.5f}, SL: {signal['stop_loss']:.5f}, TP: {signal['take_profit']:.5f}, Confidence: {signal['confidence']}%", 
                           "green" if signal['action'] == 'BUY' else "red")
            
            # Sleep for the configured interval
            cprint(f"\n⏳ Sleeping for {SLEEP_BETWEEN_RUNS_MINUTES} minutes...", "white", "on_blue")
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)
        
        except KeyboardInterrupt:
            cprint("\n👋 CRT Agent shutting down gracefully...", "white", "on_blue")
            break
        
        except Exception as e:
            cprint(f"\n❌ Error: {str(e)}", "white", "on_red")
            cprint("🔧 Moon Dev suggests checking the logs and trying again!", "white", "on_blue")
            time.sleep(300)  # Sleep for 5 minutes on error

if __name__ == "__main__":
    main()
