"""
🔄 CRT Integration Module

This module handles the integration between the CRT agent and the trading agent.
It allows CRT signals to be used as a data source for the trading agent.
"""

import logging
from src.agents.crt_agent import CRTAgent
from src.config.crt_config import CRT_PROVIDE_SIGNALS, CRT_MIN_CONFIDENCE, CRT_MIN_RISK_REWARD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CRTIntegration:
    def __init__(self):
        """Initialize the CRT integration"""
        self.crt_agent = CRTAgent()
        logger.info("CRT Integration initialized")
    
    def get_crt_signals(self):
        """Get CRT signals for the trading agent"""
        if not CRT_PROVIDE_SIGNALS:
            logger.info("CRT signal provision is disabled")
            return {}
        
        # Run the CRT agent to get signals
        signals = self.crt_agent.run()
        
        # Filter signals based on confidence and risk-reward
        filtered_signals = self._filter_signals(signals)
        
        logger.info(f"Providing {self._count_signals(filtered_signals)} CRT signals to trading agent")
        return filtered_signals
    
    def _filter_signals(self, signals):
        """Filter signals based on confidence and risk-reward"""
        filtered = {}
        
        for symbol, symbol_signals in signals.items():
            filtered[symbol] = []
            
            for signal in symbol_signals:
                # Check confidence
                if signal['confidence'] < CRT_MIN_CONFIDENCE:
                    continue
                
                # Check risk-reward
                if signal['risk_reward'] < CRT_MIN_RISK_REWARD:
                    continue
                
                # Add to filtered signals
                filtered[symbol].append(signal)
            
            # Remove empty symbols
            if not filtered[symbol]:
                filtered.pop(symbol)
        
        return filtered
    
    def _count_signals(self, signals):
        """Count the total number of signals"""
        count = 0
        for symbol, symbol_signals in signals.items():
            count += len(symbol_signals)
        return count
    
    def format_for_trading_agent(self, signals):
        """Format CRT signals for the trading agent"""
        formatted = {}
        
        for symbol, symbol_signals in signals.items():
            formatted[symbol] = {
                'strategy_signals': []
            }
            
            for signal in symbol_signals:
                formatted_signal = {
                    'strategy': 'CRT',
                    'signal_type': signal['signal_type'],
                    'action': signal['action'],
                    'entry': signal['entry'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'confidence': signal['confidence'],
                    'risk_reward': signal['risk_reward'],
                    'time': signal['time']
                }
                
                formatted[symbol]['strategy_signals'].append(formatted_signal)
        
        return formatted
