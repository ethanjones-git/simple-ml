import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
from sklearn.neural_network import MLPRegressor
from numpy import hstack
import os

#instantiate file local directory
path = os.path.realpath('..')
path = path + '/simple_ml'

#load classes/files
def smape(A, F):
    '''
    the data provider recomends time series prediction be evaluated in systematic mean absolute error (SMAPE). This is implimented through Numpy.

    More info on SMAPE:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error#:~:text=Symmetric%20mean%20absolute%20percentage%20error%20(SMAPE%20or%20sMAPE)%20is%20an,percentage%20(or%20relative)%20errors.

    :param A: actual values
    :param F: forcasted value
    :return: mean absolute error (lower is better)
    '''

    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def split_sequence(sequence, n_steps):
    '''
    This function modifies the numpy array for appropriate time series training.

    :param sequence: training data
    :param n_steps: the amount of data you want to predict
    :return:
    '''

    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

class TS_model():
    '''
    To keep code clean the follow class is created to contain all functions related to timeseries LSTM prediction.


    '''
    def __init__(self,train):
        '''
        instatiate and load local data

        :param train: name of training data in under /simple_ml/data
        '''
        # establishes a relative pathway
        path = os.path.realpath('..')
        path = path + '/simple_ml'

        # select either trian or test sets
        df_ = pd.read_csv(path + '/data/' + train + '.csv')
        df_['date'] = pd.to_datetime(df_['date'], format='%Y-%m-%d')

        # load supplimental economic data
        df_econ = []
        for i in df_['country'].unique():
            # loads monthly imports by country
            df__ = pd.read_csv(path + "/data/" + i + "_imports.csv")
            df__.columns = ['date', "imports"]
            df__['country'] = i

            # loads monthly consumer price index by country
            df___ = pd.read_csv(path + "/data/" + i + "_cpi.csv")
            df___.columns = ['date', 'cpi']
            df___['country'] = i

            # loads quarterly gdp
            df____ = pd.read_csv(path + "/data/" + i + "_gdp.csv")
            df____.columns = ['date', 'gdp']
            df____['country'] = i

            # combines either in one data frame
            df_out = pd.merge(df__, df___, how='outer', on=['date', 'country'])
            df_out = pd.merge(df_out, df____, how='outer', on=['date', 'country'])
            df_econ.append(df_out)

        # combines imports & CPI for all countries
        df_econ = pd.concat(df_econ)
        df_econ['date'] = pd.to_datetime(df_econ['date'], format='%Y-%m-%d')
        df_econ = df_econ.loc[(df_econ['date'].dt.year >= 2017) & (df_econ['date'].dt.year <= 2022)]
        df_econ.fillna(method='ffill', inplace=True)

        # annual electricity by country
        df_elc = pd.read_csv(path + '/data/elec_euro.csv')
        df_elc['Time'] = pd.to_datetime(df_elc['Time'])
        df_elc = df_elc.loc[df_elc['Country'].isin(['Germany', 'Belgium', 'France', 'Spain', 'Italy', 'Poland']) & (
            df_elc['Time'].dt.year.isin([2017, 2018, 2019, 2020, 2021, 2022])) & (
                                        df_elc['Product'] == 'Electricity') & (
                                    df_elc['Balance'] == 'Final Consumption (Calculated)')].drop(
            ['Balance', 'Product', 'Unit'], axis=1).rename(
            columns={'Country': 'country', 'Time': 'date', 'Value': 'elec'})

        # daily oil across Europe
        df_oil = pd.read_csv(path + '/data/europe_oil.csv')
        df_oil.columns = ['date', 'oil']
        df_oil['date'] = pd.to_datetime(df_oil['date'])

        # merge supplimental data (df_ext) to train/test data (df_)
        df_ext = pd.merge(df_elc, df_econ, how='inner', on=['date', 'country'])
        df_ext['ym'], df_['ym'] = df_ext['date'].dt.to_period('M'), df_['date'].dt.to_period('M')
        df = pd.merge(df_, df_ext, how='left', on=['ym', 'country'])
        df.drop('date_y', axis=1, inplace=True)
        df.rename(columns={'date_x': 'date'}, inplace=True)
        df = pd.merge(df, df_oil, how='left', on='date')
        df = df.replace('.', np.nan)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df['oil'] = pd.to_numeric(df['oil'])

        self.df = df
    def get_df(self):
        '''
        Allows you to get the dataframe from inital instantiation
        :return:
        '''
        return self.df
    def decomp(self,product, store, country,frac):
        '''
        creates training data decomposition

        :param product: product
        :param store: store
        :param country: country
        :param frac: fraction for lowess
        :return:
        '''
        df = self.df

        #conditions allows subdivision on list and indiviual value for each criterion
        if product == "":
            product = df['product'].unique().tolist()
        if store == "":
            store = df['store'].unique().tolist()
        if country == "":
            country = df['country'].unique().tolist()

        if type(product) == str and type(store) == str and type(country) == str:
            df_r = df.loc[(df['product'] == product) & (df['store'] == store) & (df['country'] == country)]
        elif type(product) == list and type(store) == str and type(country) == str:
            df_r = df.loc[(df['product'].isin(product)) & (df['store'] == store) & (df['country'] == country)]
        elif type(product) == list and type(store) == list and type(country) == str:
            df_r = df.loc[(df['product'].isin(product)) & (df['store'].isin(store)) & (df['country'] == country)]
        elif type(product) == list and type(store) == list and type(country) == list:
            df_r = df.loc[(df['product'].isin(product)) & (df['store'].isin(store)) & (df['country'].isin(country))]
        elif type(product) == str and type(store) == list and type(country) == list:
            df_r = df.loc[(df['product'] == product) & (df['store'].isin(store)) & (df['country'].isin(country))]
        elif type(product) == str and type(store) == str and type(country) == list:
            df_r = df.loc[(df['product'] == product) & (df['store'] == store) & (df['country'].isin(country))]
        elif type(product) == str and type(store) == list and type(country) == str:
            df_r = df.loc[(df['product'] == product) & (df['store'].isin(store)) & (df['country'] == country)]
        elif type(product) == list and type(store) == str and type(country) == list:
            df_r = df.loc[(df['product'].isin(product)) & (df['store'] == store) & (df['country'].isin(country))]

        #isolates hoilday spikes by any value above mean + 1 SD in January and December
        dt_lst = []
        for i in df_r['date'].dt.year.unique():
            df_ry = df_r.loc[df_r['date'].dt.year == i]
            c_mean, c_sd = np.mean(
                df_ry.loc[(df_ry['date'].dt.month == 12) | (df_ry['date'].dt.month == 1)]['num_sold']), np.std(
                df_ry.loc[(df_ry['date'].dt.month == 12) | (df_ry['date'].dt.month == 1)]['num_sold'])
            df_ry.loc[(((df_ry['date'].dt.month == 12) | (df_ry['date'].dt.month == 1)) & (
                    df_ry['num_sold'] > c_mean + c_sd)), 'num_sold'] = c_mean
            dt_lst.append(df_ry)

        #create a dataframe with removed spikes for smoothing
        ap_data = pd.concat(dt_lst)
        ap_data.reset_index(inplace=True)
        ap_data.drop(columns=['row_id', 'index'], inplace=True)

        #fit a lowess curve
        if frac != 'nofrac':
            lowess = sm.nonparametric.lowess(ap_data['num_sold'], ap_data.index, frac=frac)
            ap_data['smooth'] = pd.DataFrame(lowess)[1]

        #merge back to dataframe
        df_r.reset_index(inplace=True)
        df_r.drop(columns='index')
        df_r['num_sold_'] = ap_data['num_sold']
        df_r['smooth'] = pd.DataFrame(lowess)[1]

        #fit lowess for all external data
        for i in ['elec', 'imports', 'cpi', 'gdp', 'oil']:
            lowess = sm.nonparametric.lowess(df_r[i], df_r.index, frac=0.1)
            df_r[i + "_"] = pd.DataFrame(lowess)[1]

        #create decomposition
        df_r['num_sold_nrm'] = df_r['num_sold'] - df_r['smooth']
        df_r['num_sold_nrm_'] = df_r['num_sold_'] - df_r['smooth']
        df_r['xmas_bump'] = df_r['num_sold'] - df_r['num_sold_']

        # create 'month' features
        df_month = df_r.groupby(df_r['date'].dt.month).mean()['smooth'].reset_index()
        df_month.columns = ['month', 'mnth_avg']
        df_r['month'] = df_r['date'].dt.month
        df_r = pd.merge(df_r, df_month, how='left', on='month')
        lowess = sm.nonparametric.lowess(df_r['mnth_avg'], df_r.index, frac=0.05)
        df_r['month_avg'] = pd.DataFrame(lowess)[1]
        df_r = df_r.set_index('date')
        df_r = df_r.groupby('date').mean()

        # create 'days' feature
        df_r['dow'] = df_r.index.weekday
        df_r['dom'] = df_r.index.day

        self.df_decomp = df_r

        return df_r
    def decomp_viz(self):
        '''
        Visualization for decomposition

        :return: visualization for decomposition
        '''
        df = self.df_decomp
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=("Average Change", "Daily Sales", "Holiday Spikes"))
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['smooth']),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['num_sold_nrm_']),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['xmas_bump']),
            row=3, col=1)
        fig.update_layout(height=600, width=800, title_text="Modelling Problems")
        fig.show()
    def trend_model(self, eps, bs, ners, n_steps, items, df):
        '''
        LSTM for smooth data

        :param eps: number of training epochs
        :param bs: batch size
        :param ners: number of nerons
        :param n_steps: number of steps backwards for training data (i.e. 1 = only retain yesterday, 7= retain the last whole week)
        :param items: features model is fit witih
        :param df: training dataframe
        :return: predictions
        '''

        #standardization
        sY_1,sX_1 = MinMaxScaler(),MinMaxScaler()

        # y standardization
        df_y=sY_1.fit_transform(np.array(df[items[0]]).reshape(-1,1))
        df[items[0]] = df_y

        # x standardization
        df[items[1:]] = sX_1.fit_transform(df[items[1:]])

        #create training dataset
        a = {}
        for i in items:
            key = i
            value = np.array(df[:'2019'][i])
            value = value.reshape((len(value), 1))
            a[key] = value
        dataset_ = hstack(a.values())

        #call function
        X, y = split_sequence(dataset_, n_steps)

        #prepare test data
        y_uni = []
        for i in range(len(y)):
            y_uni.append(y[i][0])
        y_uni = np.array(y_uni)
        y_uni = y_uni.reshape((len(y_uni), 1))

        #prepare training data
        n_features = X.shape[2]

        #define model
        model = Sequential()
        model.add(LSTM(ners, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

        #fit model
        model.fit(X, y_uni, epochs=eps, verbose=0)

        #length of predicition is set to a year
        pred_len = 366
        test_predication = []
        first_eval_batch = dataset_[-n_steps:]
        current_batch = first_eval_batch.reshape((1, n_steps, n_features))

        #prepare dataset
        items_ = items[1:]
        a_ = {}
        for i in items_:
            key_ = i
            value = np.array(df['2020'][i])
            value = value.reshape((len(value), 1))
            a_[key_] = value
        dataset__ = hstack(a_.values())

        #make predictions
        for i in range(pred_len):
            current_pred = model.predict(current_batch, batch_size=bs)

            test_predication.append(current_pred[0])

            curr_ = np.array([hstack((current_pred[0], dataset__[i]))])

            current_batch = np.append(current_batch[:, 1:, :], [curr_], axis=1)
        pred = pd.DataFrame(test_predication).dropna()
        pred = sY_1.inverse_transform(np.array(pred).reshape(-1,1))

        #X transformation
        df[items[0]]  = sY_1.inverse_transform(np.array(df_y.reshape(-1, 1)))

        #Y transformation
        df[items[1:]] = sX_1.inverse_transform(df[items[1:]])

        #temporary storage to allow to be saved
        self.m = model

        #return predictions
        return pred
    def trend_save_model(self,name):
        '''
        Save model locally
        :param name:
        :return:
        '''
        model=self.m
        model.save_weights(path + "/data/out_model_trend" +str(name))
    def trend_model_psaved(self,name, bs,ners, n_steps,items, df):
        '''
        call savesd LSTM for smooth data

        :param name: name used for saved model
        :param bs: batch size
        :param ners: number of nerons
        :param n_steps: number of steps backwards for training data (i.e. 1 = only retain yesterday, 7= retain the last whole week)
        :param items: features model is fit witih
        :param df: training dataframe
        :return: predictions
        '''

        #standardization
        sY_1,sX_1 = MinMaxScaler(),MinMaxScaler()

        #y standardization
        df_y=sY_1.fit_transform(np.array(df[items[0]]).reshape(-1,1))
        df[items[0]] = df_y

        #x standardization
        df[items[1:]] = sX_1.fit_transform(df[items[1:]])

        #create training dataset
        a = {}
        for i in items:
            key = i
            value = np.array(df[:'2019'][i])
            value = value.reshape((len(value), 1))
            a[key] = value
        dataset_ = hstack(a.values())

        #call function
        X, y = split_sequence(dataset_, n_steps)

        #prepare test data
        y_uni = []
        for i in range(len(y)):
            y_uni.append(y[i][0])
        y_uni = np.array(y_uni)
        y_uni = y_uni.reshape((len(y_uni), 1))

        #prepare training data
        n_features = X.shape[2]


        #define model
        model = Sequential()
        model.add(LSTM(ners, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

        #call model
        model.load_weights(path + "/data/out_model_trend" + str(name))

        #set prediction length to a year
        pred_len = 366
        test_predication = []
        first_eval_batch = dataset_[-n_steps:]
        current_batch = first_eval_batch.reshape((1, n_steps, n_features))

        #prepare dataset
        items_ = items[1:]
        a_ = {}
        for i in items_:
            key_ = i
            value = np.array(df['2020'][i])
            value = value.reshape((len(value), 1))
            a_[key_] = value
        dataset__ = hstack(a_.values())

        #make predictions
        for i in range(pred_len):
            current_pred = model.predict(current_batch, batch_size=bs)

            test_predication.append(current_pred[0])

            curr_ = np.array([hstack((current_pred[0], dataset__[i]))])

            current_batch = np.append(current_batch[:, 1:, :], [curr_], axis=1)

        pred = pd.DataFrame(test_predication).dropna()

        pred = sY_1.inverse_transform(np.array(pred).reshape(-1,1))

        # X transformation
        df[items[0]]  = sY_1.inverse_transform(np.array(df_y.reshape(-1, 1)))

        # Y transformation
        df[items[1:]] = sX_1.inverse_transform(df[items[1:]])


        return pred
    def ts_model(self, eps, bs, ners, n_steps, items, df):
        '''
        LSTM for spikey data

        :param eps: number of training epochs
        :param bs: batch size
        :param ners: number of nerons
        :param n_steps: number of steps backwards for training data (i.e. 1 = only retain yesterday, 7= retain the last whole week)
        :param items: features model is fit witih
        :param df: training dataframe
        :return: predictions
        '''

        #standardization
        sY_1,sX_1 = MinMaxScaler(),MinMaxScaler()

        #y transformation
        df_y=sY_1.fit_transform(np.array(df[items[0]]).reshape(-1,1))
        df[items[0]] = df_y

        #x transformation
        df[items[1:]] = sX_1.fit_transform(df[items[1:]])

        #create training dataset
        a = {}
        for i in items:
            key = i
            value = np.array(df[:'2019'][i])
            value = value.reshape((len(value), 1))
            a[key] = value
        dataset_ = hstack(a.values())

        #call function
        X, y = split_sequence(dataset_, n_steps)

        #prepare test data
        y_uni = []
        for i in range(len(y)):
            y_uni.append(y[i][0])
        y_uni = np.array(y_uni)
        y_uni = y_uni.reshape((len(y_uni), 1))

        #prepare training data
        n_features = X.shape[2]

        #define model
        model = Sequential()
        model.add(LSTM(ners, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
        # fit model
        model.fit(X, y_uni, epochs=eps, verbose=0)


        #legnth of prediction set to a year
        pred_len = 366
        test_predication = []
        first_eval_batch = dataset_[-n_steps:]
        current_batch = first_eval_batch.reshape((1, n_steps, n_features))

        #prepare dataset
        items_ = items[1:]
        a_ = {}
        for i in items_:
            key_ = i
            value = np.array(df['2020'][i])
            value = value.reshape((len(value), 1))
            a_[key_] = value
        dataset__ = hstack(a_.values())

        #make predictions
        for i in range(pred_len):
            current_pred = model.predict(current_batch, batch_size=bs)
            test_predication.append(current_pred[0])
            curr_ = np.array([hstack((current_pred[0], dataset__[i]))])
            current_batch = np.append(current_batch[:, 1:, :], [curr_], axis=1)
        pred = pd.DataFrame(test_predication).dropna()
        pred = sY_1.inverse_transform(np.array(pred).reshape(-1,1))

        #X unstandardization
        df[items[0]]  = sY_1.inverse_transform(np.array(df_y.reshape(-1, 1)))

        #y unstandardization
        df[items[1:]] = sX_1.inverse_transform(df[items[1:]])

        self.m_ts = model

        return pred
    def ts_save_model(self,name):
        '''
        Save model locally
        :param name:
        :return:
        '''
        model = self.m_ts
        model.save_weights(path + "/data/out_model_ts"+ str(name))
    def ts_model_psaved(self, name,bs,ners, n_steps,items, df):
        '''
        call savesd LSTM for smooth data

        :param name: name used for saved model
        :param bs: batch size
        :param ners: number of nerons
        :param n_steps: number of steps backwards for training data (i.e. 1 = only retain yesterday, 7= retain the last whole week)
        :param items: features model is fit witih
        :param df: training dataframe
        :return: predictions

        '''

        #standarization
        sY_1,sX_1 = MinMaxScaler(),MinMaxScaler()

        #x standardization
        df_y=sY_1.fit_transform(np.array(df[items[0]]).reshape(-1,1))
        df[items[0]] = df_y

        #y standardization
        df[items[1:]] = sX_1.fit_transform(df[items[1:]])

        #create training dataset
        a = {}
        for i in items:
            key = i
            value = np.array(df[:'2019'][i])
            value = value.reshape((len(value), 1))
            a[key] = value
        dataset_ = hstack(a.values())

        #call function
        X, y = split_sequence(dataset_, n_steps)

        #prepare test data
        y_uni = []
        for i in range(len(y)):
            y_uni.append(y[i][0])
        y_uni = np.array(y_uni)
        y_uni = y_uni.reshape((len(y_uni), 1))

        #prepare training data
        n_features = X.shape[2]

        #define model
        model = Sequential()
        model.add(LSTM(ners, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

        #call model
        model.load_weights(path + "/data/out_model_ts"+ str(name))

        #predict for a year
        pred_len = 366
        test_predication = []
        first_eval_batch = dataset_[-n_steps:]
        current_batch = first_eval_batch.reshape((1, n_steps, n_features))

        #prepare dataset
        items_ = items[1:]
        a_ = {}
        for i in items_:
            key_ = i
            value = np.array(df['2020'][i])
            value = value.reshape((len(value), 1))
            a_[key_] = value
        dataset__ = hstack(a_.values())

        #make predictions
        for i in range(pred_len):
            current_pred = model.predict(current_batch, batch_size=bs)

            test_predication.append(current_pred[0])

            curr_ = np.array([hstack((current_pred[0], dataset__[i]))])

            current_batch = np.append(current_batch[:, 1:, :], [curr_], axis=1)

        pred = pd.DataFrame(test_predication).dropna()

        pred = sY_1.inverse_transform(np.array(pred).reshape(-1,1))

        #X transformation
        df[items[0]]  = sY_1.inverse_transform(np.array(df_y.reshape(-1, 1)))

        #Y transformation
        df[items[1:]] = sX_1.inverse_transform(df[items[1:]])

        return pred
    def perf_vis(self,pred,obs,df):
        y_hat_smooth=pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[obs], x=df.index, name='train'))
        fig.add_trace(go.Scatter(y=np.concatenate(y_hat_smooth), x=df['2020'].index, name='prediction'))
        fig.add_trace(go.Scatter(y=df[obs]['2020'], x=df['2020'].index, name='observation'))
        fig.show()

'''
EDA app - https://eda-jones2022.herokuapp.com/
'''

''' TRAIN ALL MODELS '''
#instatiate class with the training file
d1=TS_model('train')

''' Model 1: Trend Sales based on Germany and Belgium across country and Market'''
#list variables
y_hat_ts = []
mark=[]
prod=[]
date=[]

#countries trends will be based on
c = ['Belgium','Germany']

#call a 0.1 degree lowess curve
f=0.1

#iterate for diff market and product
for i in ['KaggleMart','KaggleRama']:
    for ii in ['Kaggle for Kids: One Smart Goose','Kaggle Advanced Techniques','Kaggle Getting Started','Kaggle Recipe Book']:
        df=d1.decomp(product=ii,store=i,country=c,frac = f)
        y_hat = d1.trend_model(eps=50, bs=1, n_steps=31, ners=30, items=['smooth', 'month'], df=df)
        y_hat_ts.append(y_hat)
        mark.append([i]*366)
        prod.append([ii]*366)
        date.append(df['2020'].index)

#create trend dataframe
df_ = pd.DataFrame(y_hat_ts)
df_['mark_'] = [item for sublist in mark for item in sublist]
df_['prod_'] = [item for sublist in prod for item in sublist]
df_['date_'] = [item for sublist in date for item in sublist]


''' Model 2: Daily Sales unique to each Product, Market, and Country'''
#list variables
y_hat_day = []
mark_=[]
prod_=[]
country_ =[]
date_ = []

#iterate thorugh diff market, product, and country
for i in ['KaggleMart','KaggleRama']:
    for ii in ['Kaggle for Kids: One Smart Goose','Kaggle Advanced Techniques','Kaggle Getting Started','Kaggle Recipe Book']:
        for iii in ['France', 'Italy', 'Belgium', 'Germany', 'Poland', 'Spain']:
            df = d1.decomp(product=ii, store=i, country=iii, frac=f)
            y_hat = d1.ts_model(eps=3000, bs=50, n_steps=1, ners=30, items=['num_sold_nrm_', 'dow', 'month', 'dom'],df=df)
            y_hat_day.append(y_hat)
            mark_.append([i] * 366)
            prod_.append([ii] * 366)
            country_.append([iii]*366)
            date_.append(df['2020'].index)

#create timeseries dataframe
df__ = pd.DataFrame(np.concatenate(y_hat_day))
df__['mark_'] = [item for sublist in mark_ for item in sublist]
df__['prod_'] = [item for sublist in prod_ for item in sublist]
df__['country_'] = [item for sublist in country_ for item in sublist]
df__['date_'] = [item for sublist in date_ for item in sublist]

''' Model 3: Holiday Spikes '''
#list variables
y_hat_holiday_ = []
mark__=[]
prod__=[]
country__=[]
date__=[]

#iterate thorugh diff market, product, and country
for i in ['KaggleMart','KaggleRama']:
    for ii in ['Kaggle for Kids: One Smart Goose','Kaggle Advanced Techniques','Kaggle Getting Started','Kaggle Recipe Book']:
        for iii in ['France', 'Italy', 'Belgium', 'Germany', 'Poland', 'Spain']:
            ''' Model 3'''
            df = d1.decomp(product=ii, store=i, country=iii, frac=f)

            # only dealing with the holiday months
            df_holi = df[(df.index > '2017-12-26') & (df.index < '2018-01-02') | (df.index > '2018-12-26') & (df.index < '2019-01-02') | (df.index > '2019-12-26') & (df.index < '2020-01-02') | (df.index > '2020-12-26')]

            # unique scalers
            sX_3 = MinMaxScaler()
            sY_3 = MinMaxScaler()

            # scale
            df_holi[['mnth_avg', 'dow', 'dom', 'month']] = sX_3.fit_transform(
                df_holi[['mnth_avg', 'dow', 'dom', 'month']])
            df_holi[['xmas_bump']] = sY_3.fit_transform(df_holi[['xmas_bump']])

            # train test sets
            X_train, y_train = df_holi[['mnth_avg', 'dow', 'dom', 'month']][:'2018'], df_holi[['xmas_bump']][:'2018']
            y_train = np.ravel(y_train)

            X_test, y_test = df_holi[['mnth_avg', 'dow', 'dom', 'month']]['2019':'2020'], df_holi[['xmas_bump']][
                                                                                          '2019':'2020']
            y_test = np.ravel(y_test)

            # best performing model is mlpRegression (neural network)
            #smape_score = make_scorer(smape, greater_is_better=False)
            model = MLPRegressor(hidden_layer_sizes=200, solver='lbfgs')
            model.fit(X_train, y_train)

            # prediction
            y_hat_holiday = model.predict(X_test)

            # unstandardize
            df_holi[['mnth_avg', 'dow', 'dom', 'month']] = sX_3.inverse_transform(df_holi[['mnth_avg', 'dow', 'dom', 'month']])

            y_hat_holiday = sY_3.inverse_transform(y_hat_holiday.reshape(-1, 1))
            df_holi[['xmas_bump']] = sY_3.inverse_transform(df_holi[['xmas_bump']])
            y_test = sY_3.inverse_transform(y_test.reshape(-1, 1))

            y_hat_holiday_.append(y_hat_holiday)
            mark__.append([i] * 12)
            prod__.append([ii] * 12)
            country__.append([iii] * 12)
            date__.append(X_test.index)

#create dataframe
df___ = pd.DataFrame(np.concatenate(y_hat_holiday_))
df___['mark_'] = [item for sublist in mark__ for item in sublist]
df___['prod_'] = [item for sublist in prod__ for item in sublist]
df___['country_'] = [item for sublist in country__ for item in sublist]
df___['date_'] = [item for sublist in date__ for item in sublist]


''' Merge all predictions into a final dataframe '''
#rename the key variable
df_.rename(columns={0:'y_hat_trend'},inplace=True)
df__.rename(columns={0:'y_hat_ts'},inplace=True)
df___.rename(columns={0:'y_hat_holi'},inplace=True)

#merge all three DFs
df_m=pd.merge(df__,df___,how='left',on=['mark_','prod_','country_','date_'])
df_m=pd.merge(df_m,df_,how='left',on=['mark_','prod_','date_'])
df_m.fillna(0,inplace=True)

#store as one dataframe
df_m.to_csv(path+'/data/pred.csv',index_label=False)

#seperate by country for github GIST storeage
df_pred_fgb=df_m.loc[df_m['country_'].isin(['France','Germany','Belgium'])]
df_pred_fgb.to_csv(path+'/data/df_pred_fgb.csv',index=False)

#store seperate dataframes
df_pred_isp=df_m.loc[df_m['country_'].isin(['Poland','Italy','Spain'])]
df_pred_isp.to_csv(path+'/data/df_pred_isp.csv',index=False)