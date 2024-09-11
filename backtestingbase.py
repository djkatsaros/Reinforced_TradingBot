#
# Event based backtesting class
# -- Base class 
#
# 

class BacktestingBase:
    def __init__(self, env, model, amount, ptc, ftc, verbose=False):
        self.env = env
        self.model = model
        self.initial_amount = amount
        self.current_balance = amount
        self.ptc = ptc
        self.ftc = ftc
        self.verbose = verbose
        self.units = 0
        self.trades = 0
        
    ########################################################
    #   Methods to get/print properties of the given bar
    ########################################################

    def get_date_price(self, bar):
        ''' Returns date and price for given bar
        '''
        date = str(self.env.data.index[bar])[:10]
        price = self.env.data[self.env.symbol].iloc[bar]
        return date, price

    def print_balance(self, bar):
        ''' Print current cash balance for bar
        '''
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance = {self.current_balance:.2f}')

    def calculate_net_wealth(self, price):
        return self.current_balance + self.units * price

    def print_net_wealth(self, bar):
        ''' Prints the net wealth (cash + position) for bar
            
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        print(f'{date} | net wealth = {net_wealth:.2f}')

    ########################################################
    #   Methods to place buy/sell/close out orders
    ########################################################
  
    def place_buy_order(self, bar, amount=None, units=None):
        ''' Places a buy order for a given bar and
            quantity of units.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
            # units = amount / price  # alternative handling
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc # adjust balance. Acct for transact costs
        self.units += units # adjust units
        self.trades += 1 # track number of trades
        if self.verbose:
            print(f'{date} | buy {units} units for {price:.4f}')
            self.print_balance(bar)

    def place_sell_order(self, bar, amount=None, units=None):
        '''  Places a sell order for a given bar and
            quantity of units.
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            units = int(amount / price)
            # units = amount / price  # altermative handling
        self.current_balance += (1 - self.ptc) * units * price - self.ftc # adjust balance. Acct for transact costs
        self.units -= units # adjust units
        self.trades += 1 # track number of trades 
        if self.verbose:
            print(f'{date} | sell {units} units for {price:.4f}')
            self.print_balance(bar)

    def close_out(self, bar):
        ''' Closes out any open position at bar.
        '''
        date, price = self.get_date_price(bar)
        print(50 * '=')
        print(f'{date} | *** CLOSING OUT ***')
        if self.units < 0:
            self.place_buy_order(bar, units= -self.units)
        else:
            self.place_sell_order(bar, units= self.units)
        if not self.verbose:
            print(f'{date} | current balance = {self.current_balance:.2f}')
        perf = (self.current_balance / self.initial_amount - 1) * 100
        print(f'{date} | net performance [%] = {perf:.4f}')
        print(f'{date} | number of trades [#] = {self.trades}')
        print(50 * '=')

        
