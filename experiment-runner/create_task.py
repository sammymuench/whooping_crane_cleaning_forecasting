"""
Given bird data, create modeling task suitable for BPR calculatior
"""

import pandas as pd
import numpy as np
from pandas import IndexSlice as idx
from collections import namedtuple


DataSet = namedtuple('DataSplit', ['x', 'y', 'info'])


def make_outcome_history_feat_names(W, outcome_col='counts'):
    return ['prev_%s_%02dback' % (outcome_col, W - ww) for ww in range(W)]

def determine_season_year(date):
    return date.year if date.month >= 7 else date.year - 1

def clip_by_month(gps, first_month=11, last_month=4):
    """
    Given that gps data tracks birds all through the year, we need some way to filter the weeks that have no birds

    ARGUMENTS: 
        gps: gps dataframe we are going to cut from, by-season. 
        first_month, last_month: months that we clip out.
    RETURNS: dataframe with some all-zero weeks cut out, based on first and last month
    """
    gps.sort_index(inplace=True)
    cut_gps = gps[(gps['month'] >= first_month) | (gps['month'] <= last_month)]
    
    weekly_counts = gps[['week_name', 'counts']].groupby(gps.index.get_level_values(1)).agg({'week_name': 'first', 'counts': 'sum'})        
    weekly_counts['start_date'] = pd.to_datetime(weekly_counts['week_name'].str.split('_to_').str[0])
    weekly_counts['season_year'] = weekly_counts['start_date'].apply(determine_season_year)
    weekly_counts = weekly_counts[weekly_counts['week_name'].isin(cut_gps['week_name'].unique())]
    return cut_gps, weekly_counts


def clip_by_sightings(gps, type_='', pct_thresh=0.05):
    """
    Given that gps data tracks birds all through the year, we need some way to filter the weeks that have no birds

    ARGUMENTS:
    df: gps dataframe we are going to cut from
    type_: choose from ['by_season', 'first_crane', 'percentile']:
        by_month: cut all months of data between pre-set months. Defaults are November (11) and April (4). 
        first_crane: Keep all rows between the first and last whooping crane seen in a given season, with exception for one whooping crane
            (there is a single crane documented in september 2012 with many zeros afterward)
        percentile: For each distribution of cranes per-day in each season, only keep rows between the first and last instance 
            higher than some pct_thresh% of the mean birds seen per-day in that season. Seasons here are determined by all rows between 
            the first and last whooping crane seen in a given year (from july-july).

    RETURNS: dataframe with some all-zero weeks cut out, depending on type_
    """
    gps.sort_index(inplace=True)

    if type_ not in ['first_crane', 'percentile']:
        raise ValueError('please read documentation for type_')
    
    if type_ == 'first_crane':
        
        # create concept of "seasons"
        weekly_counts = gps[['week_name', 'counts']].groupby(gps.index.get_level_values(1)).agg({'week_name': 'first', 'counts': 'sum'})        
        weekly_counts['start_date'] = pd.to_datetime(weekly_counts['week_name'].str.split('_to_').str[0])
        weekly_counts['season_year'] = weekly_counts['start_date'].apply(determine_season_year)

        # group by season, take a forward and backward cumsum, and filter by where both are more than say, 5
        def cumsums(season_group):
            season_group['cumsum_forward'] = season_group['counts'].cumsum() #.cumsum()
            season_group['cumsum_backward'] = season_group['counts'][::-1].cumsum()[::-1] 
            return season_group[(season_group['cumsum_forward'] > 1) & (season_group['cumsum_backward'] > 1)][['week_name', 'season_year']] # .drop(['cumsum_forward', 'cumsum_backward'], axis=1)

        new_gps = weekly_counts.groupby('season_year').apply(cumsums)
        return new_gps
    
    # TODO implement
    elif type_ == 'percentile':
        pass
        # same process as above but find mean count for each season and use that as the threshold


def enum_geoid(all_df):
    """
    Given raw df with points in boxes, set up data for modeling
    """
    if not ('week_name' in all_df.columns and 'lat' in all_df.columns and 'long' in all_df.columns):
        raise ValueError('need week_name, lat, and long in columns')

    all_df['geoid'] = all_df.apply(lambda row: (row['lat'], row['long']), axis=1)
    unique_geoids = {geoid: index for index, geoid in enumerate(all_df['geoid'].unique())}
    all_df['geoid'] = all_df['geoid'].map(unique_geoids)
    all_df['year'] = pd.to_numeric(all_df['week_name'].apply(lambda w: w.split('-')[0]))
    all_df['month'] = pd.to_numeric(all_df['week_name'].apply(lambda w: w.split('-')[1])) 
    return all_df


def create_context_df(x_df, y_df, info_df,
        first_year, last_year,
        context_size, lag_tsteps,
        year_col='year', timestep_col='timestep', outcome_col='deaths', debug=False):
    
    """
    Create individual train, valid, test dfs
    """
    new_col_names = make_outcome_history_feat_names(context_size, outcome_col=outcome_col) 
    assert last_year >= first_year

    xs = []
    ys = []
    infos = []

    for eval_year in range(first_year, last_year + 1):

        t_index = info_df[info_df[year_col] == eval_year].index
        timesteps_in_year = t_index.unique(level=timestep_col).values
        timesteps_in_year = np.sort(np.unique(timesteps_in_year))
        if debug == True: 
            print(f't index for {eval_year}', t_index)
            print(f'tsteps in year for {eval_year}', timesteps_in_year)

        for tt, tstep in enumerate(timesteps_in_year):
            
            # Make per-tstep dataframes
            x_tt_df = x_df.loc[idx[:, tstep], :].copy()
            y_tt_df = y_df.loc[idx[:, tstep], :].copy()
            full_context_size = len(y_tt_df) * context_size
            info_tt_df = info_df.loc[idx[:, tstep], :].copy()

            xhist_N = y_df.loc[idx[:, tstep-(context_size+lag_tsteps-1):(tstep-lag_tsteps)], outcome_col].values.copy()
            if len(xhist_N) < full_context_size:  # check if we have achieved the amount of context we need
                if debug: print(f'cannot use {tstep}')
                continue
            else:
                if debug: print(f'usable tstep {tstep}')

            x_hist_num_obs = xhist_N.shape[0]
            obs_per_context = x_hist_num_obs // context_size
            xhist_context = xhist_N.reshape((obs_per_context, context_size))

            for ctxt in range(context_size):
                x_tt_df[new_col_names[ctxt]] = xhist_context[:, ctxt]

            xs.append(x_tt_df)
            ys.append(y_tt_df)
            infos.append(info_tt_df)

    if len(xs) == 0:
        raise ValueError('dataset passed in to create_context_df does not have enough timesteps to adequately provide context')

    return DataSet(pd.concat(xs), pd.concat(ys), pd.concat(infos))


def create_task(
    study='gps',
    season_clip_method='first_crane',
    tr_years = [2009, 2010, 2011, 2012],
    va_years = [2013, 2014],
    te_years = [2015, 2016],
    lag_tsteps = 1,
    context_size = 3,
    geography_col = 'geoid',
    timestep_col = 'week_id',
    outcome_col = 'counts',
    year_col='season_id_year'):

    """
    overall function that creates the task
    """
    if study == 'gps':
        filepath = '../gps/2009_to_2016_weekly_500M/gps_2009_to_2016_weekly_500M_final.csv'
        tr_years = [2009, 2010, 2011, 2012],
        va_years = [2013, 2014],
        te_years = [2015, 2016]
    elif study == 'asurv':
        filepath = '../asurv/1992_to_2011_weekly_500M/asurv_1992_to_2011_weekly_500M_final.csv'
        tr_years = range(1991, 2002),
        va_years = range(2002, 2005),
        te_years = range(2005, 2008)

    gps = pd.read_csv(filepath)
    gps = enum_geoid(gps)
    
    # function args
    x_cols = [timestep_col]
    y_cols = [outcome_col]
    info_cols = [year_col]
    tr_years = np.sort(np.unique(np.asarray(tr_years)))
    va_years = np.sort(np.unique(np.asarray(va_years)))
    te_years = np.sort(np.unique(np.asarray(te_years)))
    assert np.max(tr_years) < np.min(va_years)
    assert np.max(va_years) < np.min(te_years)

    # Create the multiindex, reinserting timestep as a col not just index
    gps = gps.astype({geography_col: np.int64, timestep_col: np.int64})
    gps = gps.set_index([geography_col, timestep_col])
    gps[timestep_col] = gps.index.get_level_values(timestep_col)
    # geoid_key_df = gps.droplevel(1, axis=0)[['lat', 'long']]
    # geoid_key_df = geoid_key_df.loc[~info_df.index.duplicated(keep='first')]

    # clip weeks with zeros
    if season_clip_method == 'by_season':
        gps, valid_weeks = clip_by_month(gps, first_month=11, last_month=4)
    else:
        print(f'clipping zero weeks by {season_clip_method}')
        valid_weeks = clip_by_sightings(gps, type_=season_clip_method, pct_thresh=0.05) # first_month=11, last_month=4,
        gps = gps[gps['week_name'].isin(valid_weeks['week_name'].unique())]

    # map season year names to weeks
    week_to_season = {w: s for w, s in zip(valid_weeks['week_name'], valid_weeks['season_year'])}
    gps['season_id_year'] = gps['week_name'].map(week_to_season)

    # start x/y split
    x_df = gps[x_cols].copy()
    y_df = gps[y_cols].copy()
    info_df = gps[info_cols].copy()

    tr_tup = create_context_df(x_df, y_df, info_df,
        tr_years[0], tr_years[-1],
        context_size, lag_tsteps,
        year_col=year_col, timestep_col=timestep_col, outcome_col=outcome_col)
    
    va_tup = create_context_df(x_df, y_df, info_df,
        va_years[0], va_years[-1],
        context_size, lag_tsteps,
        year_col=year_col, timestep_col=timestep_col, outcome_col=outcome_col)

    te_tup = create_context_df(x_df, y_df, info_df,
        te_years[0], te_years[-1],
        context_size, lag_tsteps,
        year_col=year_col, timestep_col=timestep_col, outcome_col=outcome_col)

    return tr_tup, va_tup, te_tup