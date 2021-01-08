import pandas as pd
import alpaca_trade_api as tradeapi
import statistics
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import math 

# API Information
api_key = ''
api_secret = ''
base_url = 'https://paper-api.alpaca.markets'

#---------------------------------------------------------------------------------------------------------------------------

# INPUTS
risk_free_rate = .00089
chunkSize = 20
diversify = False
filename = 'companylist.csv'
#---------------------------------------------------------------------------------------------------------------------------

# instantiate REST API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

@sleep_and_retry
@limits(calls=199, period=60)
# Feed this up to n symbols at a time
def fetchBulk(symbols):
    data = api.get_barset(symbols, 'day', limit=10).df

    return data

# Get all the historical data and return the dataframe, this should be run in a loop, looping through the data output from the function above
def getHistoricalData(data, symbol):

    #data = api.get_barset(symbol, 'day', limit=10)

    info = data[symbol]['close']

    info = info.sort_index(ascending=True)

    return info

# Calculate the total change between the last value and the first value in the dataset
def calcTotalChange(historicalDF):

    most_recent_price = historicalDF[-1]

    oldest_price = historicalDF[0]

    percent_change = (most_recent_price - oldest_price) / oldest_price

    return percent_change

# Calculate the average return and standard deviation for a stock
def calcMetrics(historicalDF):

    avg = 0
    changes = []

    for i,x in enumerate(historicalDF):

        if i == 0:
            next
        
        else:
            # Percent change between each item in the list
            change = (historicalDF[i] - historicalDF[i-1]) / historicalDF[i-1]

            # Add to the changes list
            changes.append(change)

            # Keep a running average amount that we keep adding up to
            avg += change

    avg = avg / len(historicalDF)

    standard_dev = statistics.stdev(changes)

    last_price = historicalDF[-1]

    return last_price, changes, avg, standard_dev

# Combine all the functions to make calling it easier
def runAnalysis(info):

    last_price, changes, avg, stdev = calcMetrics(info)

    return last_price, changes, avg, stdev

# Creating functions to aggregate and work on multiple stocks after we have all of our data

# Create a correlation matrix for us using the output data
def createCorrelationMatrix(output):
    df = pd.DataFrame(columns = list(output.keys()))

    for x in output:
        df[x] = output[x]['changes']

    correlation_df = df.corr()

    return correlation_df

# Easier way to calculate the correlation between two stocks
def calcCorrelation(corrdf, stock1, stock2):

    result = corrdf.loc[stock1, stock2]

    return result

# Create all possible portfolios based off our correlation matrix, in order to properly diversify cross correlation of stocks
def createPortfolios(corrdf, diversify):

    portfolios = []

    diversify_threshold = 0.15

    non_diversify_threshold = .60

    rows = list(corrdf.index.values)

    for stock in list(corrdf.columns):

        tmp = []

        tmp.append(stock)

        # We want all correlations lower than our threshold so we have a fully diversified list
        for i, correlation in enumerate(corrdf[stock]):
            
            # If diversify, we want correlations lower than threshold, otherwise we want correlations higher than the threshold
            if diversify:
                if abs(correlation) <= diversify_threshold:

                    portfolio_stock = rows[i]

                    tmp.append(portfolio_stock)
            else:
                if abs(correlation) >= non_diversify_threshold:

                    portfolio_stock = rows[i]

                    tmp.append(portfolio_stock)


        portfolios.append(tmp)

    portfolios = [list(item) for item in set(tuple(row) for row in portfolios)]

    return portfolios

# Calculate weights for each position based off our buying power, so we know how many shares we can afford
def calcWeights(portfolio, buying_power, output):

    weights_info = {}

    # Evenly distribute the portfolio

    # If there are 20 stocks in our portfolio, we want each one to be weighed out at 5%
    weight = 1 / len(portfolio)

    # Round down how much cash we are allocating so we don't over allocate
    cash_allocated = math.floor(buying_power * weight)

    # Now we need to see how many shares of each stock we can afford with our cash allocated
    for stock in portfolio:

        last_price = output[stock]['last_price']

        # If we can't afford the stock, then next
        if last_price > cash_allocated:
            next
        
        else:
            # Divide the cash allocated by the stock price and round down to the nearest whole share, that will be how many we buy
            shares_to_purchase = math.floor(cash_allocated / last_price)

            weights_info[stock] = shares_to_purchase


    return weights_info

# Calculate the sharpe ratio on a portfolio
def calcSharpeRatio(portfolio, risk_free_rate, output, diversify):

    # Assuming that the portfolio is evenly distributed, calculate the average return and stdev for each stock and then calc portfolio return from there as the average of the averages

    avg_return = 0
    std_dev = 0

    # If we want to diversify, calculate the sharpe ratio, otherwise, calculate the highest average return, as we are throwing std dev out the window
    if diversify:

        for stock in portfolio:

            avg_return += output[stock]['avg_return']
            std_dev += output[stock]['stdev']

        avg_return = avg_return / len(portfolio)
        std_dev = std_dev / len(portfolio)

        sharpe_ratio = (avg_return - risk_free_rate) / std_dev

        return sharpe_ratio

    # Calculate the highest average return since we do not care about diversification
    else:

        for stock in portfolio:
            avg_return += output[stock]['avg_return']

        avg_return = avg_return / len(portfolio)

        return avg_return
    
def createOutput(chunkSize, filename):

    dfInfo = pd.read_csv(filename)

    tickers = list(set(dfInfo['Symbol']))

    output = {}

    # Create list of lists
    n = chunkSize
    tickers = [tickers[i:i+n] for i in range(0, len(tickers), n)]

    # For each chunk, fetch our bulk output
    for chunk in tqdm(tickers):

        bulkOutput = fetchBulk(chunk)

        # For each stock in our bulk output, run it's own analysis and add it to the output list
        for x in chunk:

            info = getHistoricalData(bulkOutput, x)

            last_price, changes, avg_return, stdev = runAnalysis(info)

            # Start off assuming the output is valid, if it doesn't pass our checks below, it is not valid and will not be included in the final dictionary
            isValid = True

            # If a stock has nan values, we don't want it in the final dictionary
            for change in changes:
                if math.isnan(change):
                    isValid = False

            if isValid:

                output[x] = {}

                output[x]['changes'] = changes
                output[x]['avg_return'] = avg_return
                output[x]['stdev'] = stdev
                output[x]['last_price'] = last_price

    return output

# Creating functions to perform actions on the personal profile

def getBuyingPower():
    account = api.get_account()

    buying_power = float(account.buying_power)

    return buying_power

def buyStock(stock, numShares):

    api.submit_order(
        symbol = stock,
        side = 'buy', 
        type = 'market', 
        qty = str(numShares),
        time_in_force = 'day', 
    )

def liquidatePortfolio():

    api.close_all_positions()

    api.cancel_all_orders()


# Print Info
referenceDf = pd.read_csv(filename)

numStocks = len(list(set(referenceDf['Symbol'])))

print('Indexing ' + str(numStocks) + ' securities, ' + str(chunkSize) + ' stocks at a time')

print(' ')

# Liquidate portfolio to start fresh
liquidatePortfolio()

output = createOutput(chunkSize, filename)

corrdf = createCorrelationMatrix(output)

# Create all portfolios that fall under our correlation threshold
portfolios = createPortfolios(corrdf, diversify)

sharpe_ratios = []

# Calculate sharpe ratios for each portfolio
for portfolio in portfolios:

    sharpe_ratio = calcSharpeRatio(portfolio, risk_free_rate, output, diversify)

    sharpe_ratios.append(sharpe_ratio)

# Calculate weights for the portfolio with the best sharpe ratio
best_ratio = sharpe_ratios.index(max(sharpe_ratios))

if diversify:
    print("Sharpe Ratio for Portfolio: " + str(max(sharpe_ratios)))
else:
    print("Average Return for Portfolio: " + str(max(sharpe_ratios)))

portfolio = portfolios[best_ratio]

buying_power = getBuyingPower()

weights = calcWeights(portfolio, buying_power, output)

# Purchase all shares of the portfolio
for stock in weights:

    numShares = weights[stock]

    print('Purchasing ' + str(numShares) + ' shares of ' + str(stock))
    try:
        buyStock(stock, numShares)
    except:
        next





