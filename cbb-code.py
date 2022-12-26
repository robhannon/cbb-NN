import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
#!pip install -U dmba;
from dmba import regressionSummary
from sklearn import preprocessing
from sklearn.decomposition import PCA

dm = date.fromisoformat('2022-04-10')
td = timedelta(1)

stopdate = date(2021,11,15)

while(dm>stopdate):
  url1begin = "https://barttorvik.com/trank.php?year=2022&sort=&hteam=&t2value=&conlimit=All&state=All&begin=20211101&end="

  url1date = (str(dm.year) + str(dm.month).zfill(2) + str(dm.day).zfill(2))

  url3end = "&top=0&revquad=0&quad=5&venue=All&type=All&mingames=0&csv=1#"
  url1 = url1begin + url1date + url3end
  df = pd.read_csv(url1, header = None)

  df.columns = ['Team Name', 'ADJOE', 'ADJDE', 'BARTHAG', 'Record', 'b2', 'Games_Played', 'EFG%', 'EFGD%', 'FTR', 'FTRD', 'TOR', 'TORD',
                'ORB', 'DRB', 'B4', '2P%', '2P%D', '3P%', '3P%D', 'b5','b6','b7','b8','b9','b10', 'ADJ T.','b','b.1','b.2','b.3','b.4',
                'b.5','b.6','b.7','b.8','b.9']
  df[['Wins', 'Losses']] = df['Record'].str.split('–', expand = True)
  df = df.drop(columns = ['b2','B4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b', 'b.1', 'b.2','b.3', 'b.4', 'b.5', 'b.6','b.7','b.8', 'b.9'])
  df.to_csv(url1date + ".csv")
  dm = dm - td

df1 = pd.read_csv('20220302.csv')
df1.columns

scores = pd.read_csv('https://raw.githubusercontent.com/lbenz730/NCAA_Hoops/master/3.0_Files/Results/2021-22/NCAA_Hoops_Results_3_18_2022.csv')
scores.info()

scores['year'] = scores['year'].astype(str)
scores['month'] = scores['month'].astype(str).str.zfill(2)
scores['day'] = scores['day'].astype(str).str.zfill(2)
scores['month'] = scores['month']
scores['date'] = scores['year'] + scores['month'] + scores['day']
scores['spread'] =  scores['teamscore'] - scores['oppscore']
scores = scores[scores.location == 'H']
scores = scores[scores.date > '20211116']
scores[['H_ADJOE', 'H_ADJDE', 'H_BARTHAG',
       'H_Games_Played', 'H_EFG%', 'H_EFGD%', 'H_FTR', 'H_FTRD', 'H_TOR', 'H_TORD', 'H_ORB',
       'H_DRB', 'H_2P%', 'H_2P%D', 'H_3P%', 'H_3P%D', 'H_ADJ T.', 'H_Wins', 'H_Losses','A_ADJOE', 'A_ADJDE', 'A_BARTHAG',
       'A_Games_Played', 'A_EFG%', 'A_EFGD%', 'A_FTR', 'A_FTRD', 'A_TOR', 'A_TORD', 'A_ORB',
       'A_DRB', 'A_2P%', 'A_2P%D', 'A_3P%', 'A_3P%D', 'A_ADJ T.', 'A_Wins', 'A_Losses']] = ""
scores.info()

scores = scores.replace('A&M-Corpus Christi', 'Texas A&M Corpus Chris')
scores = scores.replace('Albany (NY)', 'Albany')
scores = scores.replace('Alcorn', 'Alcorn St.')
scores = scores.replace('Ark.-Pine Bluff', 'Arkansas Pine Bluff')
scores = scores.replace('Army West Point', 'Army')
scores = scores.replace('Bethune-Cookman', 'Bethune Cookman')
scores = scores.replace('Boston U.', 'Boston University')
scores = scores.replace('California Baptist', 'Cal Baptist')
scores = scores.replace('Central Ark.', 'Central Arkansas')
scores = scores.replace('Central Conn. St.', 'Central Connecticut')
scores = scores.replace('Central Mich.', 'Central Michigan')
scores = scores.replace('Charleston So.', 'Charleston Southern')
scores = scores.replace('Col. of Charleston', 'College of Charleston')
scores = scores.replace('CSU Bakersfield', 'Cal St. Bakersfield')
scores = scores.replace('CSUN', 'Cal St. Northridge')
scores = scores.replace('Detroit Mercy', 'Detroit')
scores = scores.replace('Eastern Ill.', 'Eastern Illinois')
scores = scores.replace('Eastern Ky.', 'Eastern Kentucky')
scores = scores.replace('Eastern Mich.', 'Eastern Michigan')
scores = scores.replace('Eastern Wash.', 'Eastern Washington')
scores = scores.replace('ETSU', 'East Tennessee St.')
scores = scores.replace('FGCU', 'Florida Gulf Coast')
scores = scores.replace('Fla. Atlantic', 'Florida Atlantic')
scores = scores.replace('Ga. Southern', 'Georgia Southern')
scores = scores.replace('Gardner-Webb', 'Gardner Webb')
scores = scores.replace('Grambling', 'Grambling St.')
scores = scores.replace('Kansas City', 'UMKC')
scores = scores.replace('Lamar University', 'Lamar')
scores = scores.replace('LIU', 'LIU Brooklyn')
scores = scores.replace('LMU (CA)', 'Loyola Marymount')
scores = scores.replace('Louisiana', 'Louisiana Lafayette')
scores = scores.replace('Loyola Maryland', 'Loyola MD')
scores = scores.replace('McNeese', 'McNeese St.')
scores = scores.replace('Miami (FL)', 'Miami FL')
scores = scores.replace('Miami (OH)', 'Miami OH')
scores = scores.replace('Middle Tenn.', 'Middle Tennessee')
scores = scores.replace('Mississippi Val.', 'Mississippi Valley St.')
scores = scores.replace('N.C. A&T', 'North Carolina A&T')
scores = scores.replace('N.C. Central', 'North Carolina Central')
scores = scores.replace('NC State', 'North Carolina St.')
scores = scores.replace('North Ala.', 'North Alabama')
scores = scores.replace('Northern Ariz.', 'Northern Arizona')
scores = scores.replace('Northern Colo.', 'Northern Colorado')
scores = scores.replace('Northern Ill.', 'Northern Illinois')
scores = scores.replace('Northern Ky.', 'Northern Kentucky')
scores = scores.replace('Ole Miss', 'Mississippi')
scores = scores.replace('Omaha', 'Nebraska Omaha')
scores = scores.replace('Prairie View', 'Prairie View A&M')
scores = scores.replace('Purdue Fort Wayne', 'Fort Wayne')
scores = scores.replace('Saint Francis (PA)', 'St. Francis PA')
scores = scores.replace("Saint Mary's (CA)", "Saint Mary's")
scores = scores.replace('Seattle U', 'Seattle')
scores = scores.replace('SFA', 'Stephen F. Austin')
scores = scores.replace('SIUE', 'Southern Illinois')
scores = scores.replace('South Fla.', 'South Florida')
scores = scores.replace('Southeast Mo. St.', 'Southeast Missouri St.')
scores = scores.replace('Southeastern La.', 'Southeastern Louisiana')
scores = scores.replace('Southern California', 'USC')
scores = scores.replace('Southern Ill.', 'Southern Illinois')
scores = scores.replace('Southern Miss.', 'Southern Miss')
scores = scores.replace('Southern U.', 'Southern Utah')
scores = scores.replace('St. Francis Brooklyn', 'St. Francis NY')
scores = scores.replace("St. John's (NY)", "St. John's")
scores = scores.replace('St. Thomas (MN)', 'St. Thomas')
scores = scores.replace('UConn', 'Connecticut')
scores = scores.replace('UIC', 'Illinois Chicago')
scores = scores.replace('UIW', 'Incarnate Word')
scores = scores.replace('ULM', 'Louisiana Monroe')
scores = scores.replace('UMES', 'Maryland Eastern Shore')
scores = scores.replace('UNCW', 'UNC Wilmington')
scores = scores.replace('UNI', 'Northern Iowa')
scores = scores.replace('UT Martin', 'Tennessee Martin')
scores = scores.replace('UTRGV', 'UT Rio Grande Valley')
scores = scores.replace('Western Caro.', 'Western Carolina')
scores = scores.replace('Western Ill.', 'Western Illinois')
scores = scores.replace('Western Ky.', 'Western Kentucky')
scores = scores.replace('Western Mich.', 'Western Michigan')
scores.head()


scores.dropna(subset = ['date', 'spread'])
scores = scores[scores['D1'] == 2]
for index, row in scores.iterrows():
  d = row['date']
  df = pd.read_csv(d +".csv")
  hteam = row['team']
  ateam = row['opponent']
  hteam_df = df[df['Team Name'] == hteam]
  ateam_df = df[df['Team Name'] == ateam]
  if hteam_df.empty:
    hteam_df.loc[0] = [None, None, None, None, None, None,None, None,None, None,None,None, None,None,None,None, None,None,None,None,None, None]
  if ateam_df.empty:
     ateam_df.loc[0] = [None, None, None, None, None, None,None, None,None, None,None,None, None,None,None,None, None,None,None,None,None, None]
  
  
  scores.at[index, 'H_ADJOE'] = hteam_df.iloc[0]['ADJOE']
  scores.at[index, 'H_ADJDE'] = hteam_df.iloc[0]['ADJDE']
  scores.at[index, 'H_BARTHAG'] = hteam_df.iloc[0]['BARTHAG']
  scores.at[index, 'H_Games_Played'] = hteam_df.iloc[0]['Games_Played']
  scores.at[index, 'H_EFG%'] = hteam_df.iloc[0]['EFG%']
  scores.at[index, 'H_EFGD%'] = hteam_df.iloc[0]['EFGD%']
  scores.at[index, 'H_FTR'] = hteam_df.iloc[0]['FTR']
  scores.at[index, 'H_FTRD'] = hteam_df.iloc[0]['FTRD']
  scores.at[index, 'H_TOR'] = hteam_df.iloc[0]['TOR']
  scores.at[index, 'H_TORD'] = hteam_df.iloc[0]['TORD']
  scores.at[index, 'H_ORB'] = hteam_df.iloc[0]['ORB']
  scores.at[index, 'H_DRB'] = hteam_df.iloc[0]['DRB']
  scores.at[index, 'H_2P%'] = hteam_df.iloc[0]['2P%']
  scores.at[index, 'H_2P%D'] = hteam_df.iloc[0]['2P%D']
  scores.at[index, 'H_3P%'] = hteam_df.iloc[0]['3P%']
  scores.at[index, 'H_3P%D'] = hteam_df.iloc[0]['3P%D']
  scores.at[index, 'H_ADJ T.'] = hteam_df.iloc[0]['ADJ T.']
  scores.at[index, 'H_Wins'] = hteam_df.iloc[0]['Wins']
  scores.at[index, 'H_Losses'] = hteam_df.iloc[0]['Losses']
  
  scores.at[index, 'A_ADJOE'] = ateam_df.iloc[0]['ADJOE']
  scores.at[index, 'A_ADJDE'] = ateam_df.iloc[0]['ADJDE']
  scores.at[index, 'A_BARTHAG'] = ateam_df.iloc[0]['BARTHAG']
  scores.at[index, 'A_Games_Played'] = ateam_df.iloc[0]['Games_Played']
  scores.at[index, 'A_EFG%'] = ateam_df.iloc[0]['EFG%']
  scores.at[index, 'A_EFGD%'] = ateam_df.iloc[0]['EFGD%']
  scores.at[index, 'A_FTR'] = ateam_df.iloc[0]['FTR']
  scores.at[index, 'A_FTRD'] = ateam_df.iloc[0]['FTRD']
  scores.at[index, 'A_TOR'] = ateam_df.iloc[0]['TOR']
  scores.at[index, 'A_TORD'] = ateam_df.iloc[0]['TORD']
  scores.at[index, 'A_ORB'] = ateam_df.iloc[0]['ORB']
  scores.at[index, 'A_DRB'] = ateam_df.iloc[0]['DRB']
  scores.at[index, 'A_2P%'] = ateam_df.iloc[0]['2P%']
  scores.at[index, 'A_2P%D'] = ateam_df.iloc[0]['2P%D']
  scores.at[index, 'A_3P%'] = ateam_df.iloc[0]['3P%']
  scores.at[index, 'A_3P%D'] = ateam_df.iloc[0]['3P%D']
  scores.at[index, 'A_ADJ T.'] = ateam_df.iloc[0]['ADJ T.']
  scores.at[index, 'A_Wins'] = ateam_df.iloc[0]['Wins']
  scores.at[index, 'A_Losses'] = ateam_df.iloc[0]['Losses']

scores

scores.to_csv('scores.csv')


sns.scatterplot(data = scores, x = 'H_ADJOE', y ='teamscore', hue = 'A_ADJDE')

sns.scatterplot(data = scores, x = 'A_ADJOE', y ='oppscore', hue = 'H_ADJDE')

sns.scatterplot(data = scores, x = 'H_ADJOE', y ='A_ADJDE', hue = 'spread')

sns.scatterplot(data = scores, x = 'H_BARTHAG', y ='A_BARTHAG', hue = 'spread')


colls = {'year', 'month', 'day', 'team', 'opponent', 'location', 'canceled', 'postponed', 'OT', 'D1', 'date'}
num_scores = scores.drop(colls, axis = 1)
num_scores = num_scores.astype(np.float64)
corr_scores = num_scores.corr()
plt.figure(figsize = (28,14))
sns.heatmap(np.absolute(corr_scores), annot = True,vmin = -1, vmax = 1, fmt = ".1f", cmap = 'gray_r')


selected_vars = ['spread','H_ADJOE', 'H_ADJDE', 'H_BARTHAG', 'H_EFG%', 'H_EFGD%', 'H_FTR', 'H_FTRD', 'H_TOR', 'H_TORD', 'H_ORB', 'H_DRB', 'H_2P%', 'H_2P%D', 'H_ADJ T.','A_ADJOE', 'A_ADJDE', 'A_BARTHAG', 'A_EFG%', 'A_EFGD%', 'A_FTR', 'A_FTRD', 'A_TOR', 'A_TORD', 'A_ORB', 'A_DRB', 'A_2P%', 'A_2P%D', 'A_ADJ T.']
score_pca = scores.dropna(subset = ['spread'])
score_pca = score_pca[selected_vars]
scores_pca_spread = score_pca['spread']
score_pca = score_pca.drop('spread', 1)
selected_vars = ['H_ADJOE', 'H_ADJDE', 'H_BARTHAG', 'H_EFG%', 'H_EFGD%', 'H_FTR', 'H_FTRD', 'H_TOR', 'H_TORD', 'H_ORB', 'H_DRB', 'H_2P%', 'H_2P%D', 'H_ADJ T.','A_ADJOE', 'A_ADJDE', 'A_BARTHAG', 'A_EFG%', 'A_EFGD%', 'A_FTR', 'A_FTRD', 'A_TOR', 'A_TORD', 'A_ORB', 'A_DRB', 'A_2P%', 'A_2P%D', 'A_ADJ T.']
scaled_arr = preprocessing.scale(score_pca)
scaled_df = pd.DataFrame(scaled_arr)
pcs = PCA(n_components = 8)
principalComponents = pcs.fit_transform(scaled_df)
principalDF = pd.DataFrame(
    principalComponents.round(3),
    columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
)
score_pca.reset_index(drop=True, inplace=True)
principalDF.reset_index(drop=True, inplace=True)
scores_aug = pd.concat([score_pca, principalDF], axis = 1)

scores_aug['spread'] = scores_pca_spread.to_numpy()

scores_aug

cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'spread']
scoresN = scores_aug[cols]
scoresN = scoresN.dropna(subset = ['spread'])


y_nonscaled = scoresN['spread'].astype(float)
y_nonscaled = pd.DataFrame(y_nonscaled)
x_nonscaled = scoresN.drop(columns=['spread']).astype(float)

scaleOutput = MinMaxScaler()
scaleInput = MinMaxScaler()

x = scaleInput.fit_transform(x_nonscaled)
y = scaleOutput.fit_transform(y_nonscaled)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=1)


score_nnet = MLPRegressor(hidden_layer_sizes=(5,5), activation='logistic', solver='lbfgs', random_state=1, alpha = 0.25)
score_nnet.fit(x_train, y_train)

y_pred = score_nnet.predict(x_test)

y_actual = scaleOutput.inverse_transform(y_test).ravel()
y_pred = scaleOutput.inverse_transform([score_nnet.predict(x_test)]).ravel()


regressionSummary(y_pred, y_actual)

output = pd.concat([pd.DataFrame(y_pred,columns = ['pred']),pd.DataFrame(y_actual,columns = ['actual'])], axis = 1)
output.to_csv('output.csv')
output['diff'] = abs(output['pred'] - output['actual'])

sns.scatterplot(data = output, x = 'pred', y = 'actual')

y_tot_act = scaleOutput.inverse_transform(y).ravel()
y_tot_pred = scaleOutput.inverse_transform([score_nnet.predict(x)]).ravel()

tot_out = pd.concat([pd.DataFrame(y_tot_pred,columns = ['pred']),pd.DataFrame(y_tot_act,columns = ['actual'])], axis = 1)

sns.scatterplot(data = tot_out, x = 'pred', y = 'actual')


scores_out = scores.dropna(subset = ['spread'])
scores_out.reset_index(drop=True,inplace=True)
scores_out['pred'] = pd.Series(tot_out['pred'])
scores_out['diff'] = abs(scores_out['pred'] - scores_out['spread'])
scores_out.to_csv('output.csv')
scores_out['is_correct'] = np.where(((scores_out['spread'] > 0) & (scores_out['pred'] > 0)) | ((scores_out['spread'] < 0) & (scores_out['pred'] < 0)), 1, 0)

scores_out

scores_out['total_points'] = scores_out['teamscore'] + scores_out['oppscore']
sns.scatterplot(data = scores_out, x = 'pred', y = 'spread', hue = 'total_points')

sns.scatterplot(data = scores_out, x = 'pred', y = 'spread', hue='is_correct')

sns.regplot(data = scores_out, x = 'total_points', y = 'diff')

spread_hist = plt.hist(scores_out['spread'])
plt.title('Distribution of Actual Spread')
plt.xlabel('Actual Spread')
plt.ylabel('# of occurences')

predSpred_hist = plt.hist(scores_out['pred'])
plt.title('Distribution of Predicted Spread')
plt.xlabel('Predicted Spread')
plt.ylabel('# of occurences')



scores_out['date'] = pd.to_datetime(scores_out[['year','month','day']])
plt.figure(figsize = (20,10))
plt.plot_date(scores_out['date'],scores_out['diff'])

plt.figure(figsize = (20,10))
scores_out['date_ordinal'] = pd.to_datetime(scores_out['date']).apply( lambda date: date.toordinal())
ax = sns.regplot(data = scores_out, x = 'date_ordinal', y = 'diff', scatter_kws= {'color': 'black'}, line_kws={'color': 'orange'})
ax.set_xlabel('Date')
ax.set_title('Difference between Actual and Predicted Spread Over Time')
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticklabels(new_labels)

sns.regplot(data = scores_out, x = 'diff', y = 'total_points')


scores_out['spread'].describe()

scores_out['pred'].describe()

scores_out['diff'].describe()

scores_out['is_correct'].describe()

scores_out['diff'].median()


vegas = pd.read_csv('ncaabb21.csv')
vegas.date = pd.to_datetime(vegas.date)
vegas_use = vegas[['date', 'home', 'line']]


scores_with_vegas = scores_out.merge(vegas_use, left_on = ['date', 'team'], right_on= ['date', 'home'] )
scores_with_vegas.to_csv('withvegas.csv')
scores_with_vegas


scores_with_vegas['vegas_diff'] = abs(scores_with_vegas['spread'] - scores_with_vegas['line'])
scores_with_vegas['isModelCloser'] = np.where(scores_with_vegas['vegas_diff'] > scores_with_vegas['diff'], 1, 0)
scores_with_vegas['bettingOutcome'] = np.where(scores_with_vegas['vegas_diff'] > scores_with_vegas['diff'], 0.91, -1)
scores_with_vegas['is_vegas_correct'] = np.where(((scores_with_vegas['spread'] > 0) & (scores_with_vegas['line'] > 0)) | ((scores_with_vegas['spread'] < 0) & (scores_with_vegas['line'] < 0)), 1, 0)


scores_with_vegas['vegas_diff'].describe()

scores_with_vegas['vegas_diff'].median()

scores_with_vegas['is_vegas_correct'].describe()

scores_with_vegas['isModelCloser'].describe()

scores_with_vegas['bettingOutcome'].describe()

scores_with_vegas['bettingOutcome'].sum()

scores_with_vegas['betting_overtime'] = ""
scores_with_vegas['betting_overtime'][0] = scores_with_vegas['bettingOutcome'][0]
for row in range(1,len(scores_with_vegas)):
  scores_with_vegas['betting_overtime'][row] = (scores_with_vegas['betting_overtime'][row-1]+scores_with_vegas['bettingOutcome'][row])

scores_with_vegas['betting_overtime']

ax = plt.plot(scores_with_vegas.index, scores_with_vegas['betting_overtime'])
plt.title('Betting Return Over Time Based on a 1$ Bet')
plt.ylabel('Return ($)')
plt.xlabel('Number of Games Played')

sns.boxplot(x = scores_with_vegas['diff'], )
plt.xlim(0,40)
plt.title('Difference in Spread from Model to Reality')

sns.boxplot(x = scores_with_vegas['vegas_diff'])
plt.xlim(0,40)
plt.title('Difference in Spread from Vegas to Reality')

