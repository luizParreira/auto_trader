# vim: sta:et:sw=4:ts=4:sts=4
'''
Trading Simulator
'''
import pandas as pd
import numpy as np

class TradingSimulator(object):
    """
    Trading simulator, responsible for handling the data transitions involved on each step
    of trading. It also keeps track of the data associated with the simulation.
    """
    def __init__(self, value, prices, pair, trading_cost=.0025):
        self.value = value
        self.trading_cost = trading_cost
        self.pair = pair
        self.base_symbol, self.trading_symbol = self.pair.split("_")
        self.prices = prices
        self.dates = self.prices.index
        self.trades = self._build_zero_filled_df(self.prices)
        self.holdings = self._build_zero_filled_df(self.prices)
        self.values = self._build_zero_filled_df(self.prices)
        self.portfolio_value = self._build_portfolio_value_df(self.prices)
        self.actions = np.zeros(len(self.prices.index))
        self._step = 0

    def reset(self):
        '''
        Reset simulator data
        '''
        self._step = 0
        self.trades = self._build_zero_filled_df(self.prices)
        self.holdings = self._build_zero_filled_df(self.prices)
        self.values = self._build_zero_filled_df(self.prices)
        self.portfolio_value = self._build_portfolio_value_df(self.prices)
        self.actions = np.zeros(len(self.prices.index))

    def step(self, action):
        """
        Step another day into the simulation.
        Actions:
        - BUY(1)
        - SELL(-1)
        """
        assert action in ["BUY", "SELL"]

        date = self.dates[self._step]
        previous_date = date if self._step == 0 else self.dates[self._step - 1]
        price = self.prices[self.pair][date]
        self.actions[self._step] = 1 if action == "BUY" else -1

        # Initialize trades with `value`
        if self._step == 0:
            self.trades[self.base_symbol][date] = self.value

        if action == "BUY":
            self._buy(price, previous_date, date)
        if action == "SELL":
            self._sell(price, previous_date, date)

        self._update_holdings_and_values(price, date)

        # Update portfolio value
        port_value = self.values[self.values.columns].sum(axis=1)[date]
        self.portfolio_value["portfolio_value"][date] = port_value

        # Compute the step reward
        currently_holding = self.holdings[self.trading_symbol][date] > 0
        if currently_holding:
            reward = (1 - self.trading_cost) * self.prices["daily_return"][date]
            if self._step == 0:
                reward = -self.trading_cost
        else:
            reward = (1 - self.trading_cost) * self.prices["daily_return"][date] * -1.0
        info = {
            "reward": reward,
            "portfolio_value": self.portfolio_value,
            "holdings": self.holdings,
            "currently_holding": currently_holding,
            "values": self.values,
            "actions": self.actions
        }
        self._step += 1
        return reward, info

    def _update_holdings_and_values(self, price, date):
        ts = self.trading_symbol
        bs = self.base_symbol
        # Update holdings data frame
        if self.trades[ts][date] <= 0:
            self.holdings[ts][date] = 0
            self.values[ts][date] = 0
        else:
            self.holdings[ts][date] = self.trades[ts][date]
            self.values[ts][date] = price * self.trades[ts][date]

        if self.trades[bs][date] <= 0:
            self.holdings[bs][date] = 0
            self.values[bs][date] = 0
        else:
            self.holdings[bs][date] = self.trades[bs][date]
            self.values[bs][date] = self.holdings[bs][date]
        self.value = self.holdings[bs][date]
        return

    def _sell(self, price, previous_date, date):
        '''
        Private method responsible for selling `trading_symbol` for a specific `price`
        at given `date`
        '''
        ts = self.trading_symbol
        bs = self.base_symbol
        previous_holding = self.holdings[ts][previous_date]
        if previous_holding > 0:
            self.trades[ts][date] = -previous_holding
            self.trades[bs][date] = previous_holding * price
            self.trading_cost = 0.25 / 100.0
        else:
            self.trades[ts][date] = self.trades[ts][previous_date]
            self.trades[bs][date] = self.trades[bs][previous_date]
            self.trading_cost = 0
        return

    def _buy(self, price, previous_date, date):
        '''
        Private method responsible for buying `trading_symbol` for a specific `price`
        at given `date`
        '''
        ts = self.trading_symbol
        bs = self.base_symbol
        if self._step == 0 or self.holdings[ts][previous_date] == 0:
            self.trades[ts][date] = self.value / price
            self.trades[bs][date] = -self.value
            self.trading_cost = 0.15 / 100.0
        else:
            self.trades[ts][date] = self.trades[ts][previous_date]
            self.trades[bs][date] = self.trades[bs][previous_date]
            self.trading_cost = 0
        return


    def _build_zero_filled_df(self, df_base):
        '''
        Utils method used to build a zero-filled df.
        '''
        columns = [col for col in df_base.columns if col != "daily_return"]
        _base_pair, trading_pair = self.pair.split("_")
        columns = [trading_pair if col == self.pair else col for col in columns]
        shape = np.zeros((df_base.shape[0], len(columns)))
        return pd.DataFrame(shape, columns=columns, index=self.dates)

    def _build_portfolio_value_df(self, df_base):
        '''
        Util method used to create portfolio value data frame
        '''
        return pd.DataFrame(
            np.zeros((df_base.shape[0], 1)),
            columns=["portfolio_value"],
            index=self.dates
        )
