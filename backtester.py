import pandas as pd
import numpy as np
from data_fetcher import fetch_forex_data
from technical_analysis import calculate_indicators
from signal_generator import generate_signals
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Backtester:
    def __init__(self, symbols, start_date, end_date, initial_balance=10000):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {symbol: 0 for symbol in symbols}
        self.trades = []
        self.data = {}

    def load_data(self):
        logger.info(f"Loading historical data for {self.symbols} from {self.start_date} to {self.end_date}")
        for symbol in self.symbols:
            data = fetch_forex_data(symbol, period="max", interval="1d")
            if data is not None and not data.empty:
                logger.debug(f"Raw data received for {symbol}:\n{data.head()}")
                logger.debug(f"Data shape for {symbol}: {data.shape}")
                data = data.loc[self.start_date:self.end_date]
                data = calculate_indicators(data)  # This now includes ADX calculation
                logger.debug(f"Processed data for {symbol}:\n{data.head()}")
                logger.debug(f"Processed data shape for {symbol}: {data.shape}")
                self.data[symbol] = data
            else:
                logger.error(f"Failed to load historical data for {symbol}")
                return False
        return True

    def run_backtest(self):
        if not self.data:
            logger.error("No data available for backtesting")
            return False

        logger.info("Running backtest")
        for date in self.data[self.symbols[0]].index:
            for symbol in self.symbols:
                current_data = self.data[symbol].loc[:date]
                signals = generate_signals(current_data)
                self._process_signals(symbol, signals, current_data.iloc[-1])

        return True

    def _process_signals(self, symbol, signals, current_price_data):
        for signal in signals:
            if signal['action'] == 'BUY' and self.positions[symbol] <= 0:
                self._open_position(symbol, signal, current_price_data, 'BUY')
            elif signal['action'] == 'SELL' and self.positions[symbol] >= 0:
                self._open_position(symbol, signal, current_price_data, 'SELL')

        self._check_stop_loss_take_profit(symbol, current_price_data)

    def _open_position(self, symbol, signal, current_price_data, action):
        entry_price = current_price_data['Close']
        risk_per_trade = self.balance * 0.02  # 2% risk per trade
        position_size = risk_per_trade / abs(entry_price - signal['stop_loss'])
        cost = position_size * entry_price

        if action == 'BUY':
            self.positions[symbol] = position_size
        else:
            self.positions[symbol] = -position_size

        self.balance -= cost
        self.trades.append({
            'date': current_price_data.name,
            'symbol': symbol,
            'action': action,
            'price': entry_price,
            'size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit1': signal['tp1'],
            'take_profit2': signal['tp2'],
            'take_profit3': signal['tp3']
        })

        logger.info(f"Opened {action} position for {symbol}: {position_size} units at {entry_price}")

    def _check_stop_loss_take_profit(self, symbol, current_price_data):
        if not self.trades or self.positions[symbol] == 0:
            return

        last_trade = next(trade for trade in reversed(self.trades) if trade['symbol'] == symbol and 'close_date' not in trade)
        current_price = current_price_data['Close']

        if last_trade['action'] == 'BUY':
            if current_price <= last_trade['stop_loss']:
                self._close_position(symbol, current_price, 'Stop Loss')
            elif current_price >= last_trade['take_profit3']:
                self._close_position(symbol, current_price, 'Take Profit 3')
            elif current_price >= last_trade['take_profit2']:
                self._close_position(symbol, current_price, 'Take Profit 2')
            elif current_price >= last_trade['take_profit1']:
                self._close_position(symbol, current_price, 'Take Profit 1')
        elif last_trade['action'] == 'SELL':
            if current_price >= last_trade['stop_loss']:
                self._close_position(symbol, current_price, 'Stop Loss')
            elif current_price <= last_trade['take_profit3']:
                self._close_position(symbol, current_price, 'Take Profit 3')
            elif current_price <= last_trade['take_profit2']:
                self._close_position(symbol, current_price, 'Take Profit 2')
            elif current_price <= last_trade['take_profit1']:
                self._close_position(symbol, current_price, 'Take Profit 1')

    def _close_position(self, symbol, current_price, reason):
        last_trade = next(trade for trade in reversed(self.trades) if trade['symbol'] == symbol and 'close_date' not in trade)
        profit_loss = (current_price - last_trade['price']) * self.positions[symbol]
        self.balance += (abs(self.positions[symbol]) * current_price) + profit_loss
        logger.info(f"Closed position for {symbol}: {reason} at {current_price}. Profit/Loss: {profit_loss}")

        last_trade['close_date'] = self.data[symbol].index[-1]
        last_trade['close_price'] = current_price
        last_trade['profit_loss'] = profit_loss
        last_trade['close_reason'] = reason

        self.positions[symbol] = 0

    def calculate_performance(self):
        logger.info("Calculating performance metrics")
        total_profit_loss = sum(trade['profit_loss'] for trade in self.trades if 'profit_loss' in trade)
        winning_trades = [trade for trade in self.trades if trade.get('profit_loss', 0) > 0]
        losing_trades = [trade for trade in self.trades if trade.get('profit_loss', 0) <= 0]

        performance = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_profit_loss': total_profit_loss,
            'final_balance': self.balance,
            'return_percentage': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'profit_factor': self._calculate_profit_factor(),
            'average_trade': total_profit_loss / len(self.trades) if self.trades else 0,
        }

        return performance

    def _calculate_sharpe_ratio(self):
        if not self.trades:
            return 0

        returns = [(trade['close_price'] - trade['price']) / trade['price'] for trade in self.trades if 'close_price' in trade]
        if not returns:
            return 0

        return (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized Sharpe Ratio

    def _calculate_max_drawdown(self):
        if not self.trades:
            return 0

        cumulative_returns = np.cumsum([trade.get('profit_loss', 0) for trade in self.trades])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)

    def _calculate_profit_factor(self):
        if not self.trades:
            return 0

        gross_profit = sum(trade['profit_loss'] for trade in self.trades if trade.get('profit_loss', 0) > 0)
        gross_loss = abs(sum(trade['profit_loss'] for trade in self.trades if trade.get('profit_loss', 0) < 0))

        return gross_profit / gross_loss if gross_loss != 0 else 0

    def generate_report(self):
        performance = self.calculate_performance()
        report = f"""
Backtesting Report
==================
Period: {self.start_date} to {self.end_date}
Symbols: {', '.join(self.symbols)}
Initial Balance: ${self.initial_balance:.2f}
Final Balance: ${performance['final_balance']:.2f}

Performance Metrics:
--------------------
Total Trades: {performance['total_trades']}
Winning Trades: {performance['winning_trades']}
Losing Trades: {performance['losing_trades']}
Win Rate: {performance['win_rate']:.2%}
Total Profit/Loss: ${performance['total_profit_loss']:.2f}
Return: {performance['return_percentage']:.2f}%
Sharpe Ratio: {performance['sharpe_ratio']:.2f}
Max Drawdown: {performance['max_drawdown']:.2%}
Profit Factor: {performance['profit_factor']:.2f}
Average Trade: ${performance['average_trade']:.2f}
        """
        return report

    def generate_equity_curve(self):
        equity_curve = pd.DataFrame(index=self.data[self.symbols[0]].index, columns=['Equity'])
        equity_curve['Equity'] = self.initial_balance

        for trade in self.trades:
            if 'close_date' in trade:
                equity_curve.loc[trade['close_date']:, 'Equity'] += trade['profit_loss']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Equity Curve', 'Drawdown'))

        # Equity Curve
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve['Equity'], name='Equity'), row=1, col=1)

        # Drawdown
        drawdown = (equity_curve['Equity'].cummax() - equity_curve['Equity']) / equity_curve['Equity'].cummax()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='Drawdown', fill='tozeroy'), row=2, col=1)

        fig.update_layout(height=800, title_text="Equity Curve and Drawdown")
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        return fig

def run_backtest(symbols, start_date, end_date):
    logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
    backtester = Backtester(symbols, start_date, end_date)
    if backtester.load_data():
        if backtester.run_backtest():
            report = backtester.generate_report()
            equity_curve = backtester.generate_equity_curve()
            logger.info("Backtesting completed successfully")
            return report, equity_curve
        else:
            logger.error("Failed to run backtest")
    else:
        logger.error("Failed to load data for backtesting")
    return None, None
