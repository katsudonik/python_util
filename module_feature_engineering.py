class FeatureEngineering:
        
    def main(self, start_date=None, end_date=None):
        self.check_data(self.input_ds)
        ds = self.del_bug_data(self.input_ds)
        self.output_ds = self.add_features(ds, start_date=start_date, end_date=end_date)
        return self.output_ds
    
    def set_input_ds(self, ds):
        self.input_ds = ds
    
    def load_file(self, path_input_list, input_type, encoding='utf8'):
        str_columns = [
            'a',
            'b',
            'c',
        ]
        dtype = {}
        for i in str_columns:
            dtype[i] = str

        ds = []
        for input_path in path_input_list:
            print('loading', input_path)
            if input_type == 'csv':
                media_ds = pd.read_csv(input_path, encoding=encoding, parse_dates=['date'], dtype=dtype)    
            elif input_type == 'pkl':
                media_ds = util.pickle_load(input_path)
                media_ds['date'] = pd.to_datetime(media_ds['date'])
                for column in str_columns:
                    media_ds[column] = media_ds[column].astype(str)
            else:
                raise ValueError('input type is not in (csv, pkl)')
            
            #check columns
            missings = list( set([
              'a',
              'b',
              'c',
              'd',
              'e',
            ]) - set(media_ds.columns) )
            if len(missings) > 0:
                raise ValueError('missing columns:'  + ','.join(missings))
        
            ds.append(media_ds.fillna("__BLANK__")) # replace Nan -> "__BLANK__"
        self.set_input_ds(pd.concat(ds).reset_index(drop=True)) # union all of <all media's records>
        return self.input_ds
    
    def output_file(self, path_output, output_type, encoding='utf8'):
        if output_type == 'csv':
            self.output_ds.to_csv(path_output, encoding=encoding, index=False)
        elif output_type == 'pkl':
            self.output_ds.to_pickle(path_output)

    def check_data(self, ds):
        #raise Error

    def del_bug_data(self, ds):
        ds = ds[~(ds['a'] == '__BLANK__')]
        return ds
            
    def add_features(self, ds, start_date=None, end_date=None):
        # add date's info
        ds['year'] = ds['date'].dt.year
        ds['month'] = ds['date'].dt.month
        ds['day'] = ds['date'].dt.day
        ds['yyyymm'] = (ds['year'] * 100 + ds['month'])
        ds['weekday'] = ds['date'].dt.dayofweek # monday: 0, sunday: 6

        # sampling_by_date
        if start_date is None:
            start_date = ds['date'].min()
        if end_date is None:
            end_date = ds['date'].max()
        print('start_date:', start_date)
        print('end_date:', end_date)
        ds = ds[(ds['date'] >= start_date) & (ds['date'] <= end_date)]

        # add holiday_flg
        holidays = np.array([])
        tmp_holidays = np.array(jpholiday.holidays(start_date.date(), end_date.date()))
        if len(tmp_holidays) > 0:
            holidays = tmp_holidays[:,0]
        ds['holiday_flg'] = False
        ds.loc[ ds['date'].dt.date.isin(holidays), 'holiday_flg' ] = True

        # sort
        ds = ds.sort_values(['a', 'date'])
        ds = ds.reset_index(drop=True)

        #cast
        for column in [
              'a',
              'b',
              'c',
              'd',
              'e',
        ]:
            ds[column] = ds[column].astype(str)
        return ds

    # self.input_ds: campaigns_ds
    def generate_simulation_ds_to_end_of_month(self, start_date):
        dates = util.daterange(start_date, util.end_of_month_date(start_date))
        _ds = []
        for _id in list(set(self.input_ds['campaign_id'])):
            rows = util.df_copy_row(
                self.input_ds[(self.input_ds['campaign_id'] == _id)].sample(n=1), 
                len(dates))
            rows['date'] = dates
            _ds.append(rows)
        self.input_ds = pd.concat(_ds).reset_index().sort_values(['a', 'date'])
        return self.input_ds
