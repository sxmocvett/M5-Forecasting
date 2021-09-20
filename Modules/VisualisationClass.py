import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import plotly.graph_objs as go
import pandas as pd


class Vizualization:
    """
    Визуализация зависимостей категориальных и количественных
    признаков друг от друга
    Атрибуты:
        data - датафрейм
        cat_columns - список категориальных признаков
        num_columns - список количественных признаков
        cat_combo_2 - набор уникальных пар категориальных признаков
        num_combo_2 - набор уникальных пар количественных признаков
        num_combo_3 - набор уникальных троек количественных признаков
    """
    def __init__(self, data=None, train_data=None, test_data=None):
        if data is not None:
            self.data = data
            self.cat_columns = data.select_dtypes(include='object').columns.tolist()
            self.num_columns = data.select_dtypes(exclude='object').columns.tolist()
            self.cat_combo_2 = itertools.combinations(self.cat_columns, 2)
            self.num_combo_2 = itertools.combinations(self.num_columns, 2)
            self.num_combo_3 = itertools.combinations(self.num_columns, 3)
        if (train_data is not None) and (test_data is not None):
            self.train_data = train_data
            self.test_data = test_data

    def doublecat_count_plot(self):
        """
        Построение графиков: x - классы категориального признака,
                             y - кол-во элементов, 
                             color - классы категориального признака
        """
        for cat in self.cat_combo_2:
            fig = plt.figure()
            catplot = sns.catplot(x=cat[0],
                                  hue=cat[1],
                                  data=self.data,
                                  height=5,
                                  aspect=2,
                                  kind="count")
            catplot.fig.suptitle(cat[0]+' / '+cat[1]+' / '+'count', y=1.05)
            plt.grid()
    
    def triplenum_plot(self):
        """
        Построение графиков: x, y, color - количественные признаки
        """
        for num in self.num_combo_3:
            fig = plt.figure()
            plt.figure(figsize=[12, 8], dpi=80)
            plt.scatter(x=self.data[num[0]],
                        y=self.data[num[1]],
                        c=self.data[num[2]], cmap='viridis')
            plt.title(num[0]+' / '+num[1]+' / '+num[2], y=1.05)
            plt.xlabel(num[0])
            plt.ylabel(num[1])
            plt.colorbar(label=num[2])
    
    def num_cat_dencity_plot(self):
        """
        Построение графиков: x - количестенный признак,
                             y - плотность,
                             color - классы категориального признака
        """
        for num, cat in list(product(self.num_columns, self.cat_columns)):
                plt.figure(dpi=100)
                sns.kdeplot(data=data, x=num, hue=cat, bw=.5)
                plt.title(num+' / '+cat+' / '+'dencity')
                plt.grid()
                
    def num_num_cat_plot(self):
        """
        Построение графиков: x - количестенный признак, 
                             y - количестенный признак, 
                             color - классы категориального признака
        """
        for num, cat in list(product(self.num_combo_2, self.cat_columns)):
                plt.figure(figsize=[12, 8], dpi=80)
                data_ = self.data[[num[0], num[1], cat]]
                sns.scatterplot(x=num[0],
                                    y=num[1],
                                    hue=cat,
                                    data=data_)
                plt.title(num[0]+' / '+num[1]+' / '+cat, y=1.05)

    def cat_cat_num_plot(self):
        """
        Построение графиков: x - классы категориального признака, 
                             y - количестенный признак, 
                             color - классы категориального признака
        """
        for cat, num in list(product(self.cat_combo_2, self.num_columns)):
                plt.figure()
                g = sns.catplot(x=cat[0],
                                y=num,
                                hue=cat[1],
                                data=self.data,
                                height=5,
                                aspect=2)
                g.fig.suptitle(cat[0]+' / '+cat[1] + ' / '+num, y=1.05)

    def cat_histograms(self):
        """
        Построение графиков: x - классы категориального признака, 
                             y - кол-во элементов 
        """
        plt.figure(figsize=[20, 25], dpi=70)
        i = 1
        for cat in self.cat_columns:
            ax = plt.subplot(len(self.cat_columns), 2, i)
            sns.countplot(x=cat,data=self.data)
            ax.set_xlabel(cat)
            ax.set_ylabel('Count')
            ax.set_title ('Count of categories for {}'. format(cat), fontsize = 15)
            plt.subplots_adjust(hspace = 1)
            i += 1

    def num_histograms(self):
        """
        Построение графиков: распределение значений количественного признака
        """
        plt.figure(figsize=[20, 25])
        i = 1
        for num in self.num_columns:
            ax = plt.subplot(len(self.num_columns), 1, i)
            data[num].hist(bins=50)
            ax.set_xlabel(num)
            ax.set_ylabel('Count')
            i += 1
    
    def time_moving_average_plot(self, columns, n_roll):
        """
        Построение графика временного ряда со скользщяим средним
        Атрибуты:
            columns - отображаемые временные ряды
            n_roll - величина окна
        """
        fig = go.Figure()

        for prod in columns:
            fig.add_trace(go.Scatter(x=self.data.index,
                                     y=self.data[prod],
                                     name=prod))
            fig.add_trace(go.Scatter(x=self.data.index,
                                     y=self.data[prod].rolling(window=n_roll).mean(),
                                     name=prod+'_smoothing'))
            
            fig.update_layout(yaxis_title="Sales", xaxis_title="Time",
                          title=f"Sales for {columns} with average_smoothing ({n_roll} {self.data.index.inferred_freq})",
                          template='plotly_white',
                          xaxis=dict(
                              rangeselector=dict(
                                      buttons=list([
                                          dict(count=6, label='6m',
                                               step='month', stepmode='backward'),
                                          dict(count=12, label='12m',
                                               step='month', stepmode='backward'),
                                          dict(count=18, label='18m',
                                               step='month', stepmode='backward'),
                                          dict(step='all')])), rangeslider=dict(visible=True), type='date'))
        return fig
    
    def time_average_plot(self, columns, average_period):
        """
        Построение графика временного ряда с группировкой по периоду
        Args:
            columns - отображаемые временные ряды
            average_period - период
        """
        fig = go.Figure()
        example = self.data.groupby(by=[average_period]).mean()
    
        for prod in columns:
            fig.add_trace(go.Scatter(x=example.index,
                                     y=example[prod],
                                     mode='lines',
                                     name=prod))
        
        fig.update_layout(yaxis_title="Sales",
                          xaxis_title=average_period,
                          title=f'Mean sales for {columns} per {average_period}',
                          template='plotly_white')
        return fig
    
    def time_boxplot(self, columns):
        """
        Построение коробчатой диаграммы временного ряда
        Args:
            columns - отображаемые временные ряды
        """        
        fig = go.Figure()

        for prod in columns:
            fig.add_trace(go.Box(y=self.data[prod], name=prod))
        
        fig.update_layout(yaxis_title="Sales",
                          xaxis_title='Categories',
                          title=f'Sales for {columns}',
                          template='plotly_white')
        return fig
    
    def time_exp_plot(self, columns, alpha):
        """
        Построение графика временного ряда с экспоненциальным сглаживанием
        Args:
            columns - отображаемые временные ряды
            alpha - коэффициент экспоненциального сглаживания
        """
        fig = go.Figure()

        def exponential_smoothing(series, alpha):
            result = [series[0]] # first value is same as series
            for n in range(1, len(series)):
                result.append(alpha * series[n] + (1 - alpha) * result[n-1])
            return pd.Series(result, index=series.index)
    
        for prod in columns:
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=self.data[prod], 
                                     name=prod))
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=exponential_smoothing(self.data[prod], alpha), 
                                     name=prod+f'_smoothing (a={alpha})'))

        fig.update_layout(yaxis_title="Sales", xaxis_title="Time",
                      title=f"Sales for {columns} with exp_smoothing",
                      template='plotly_white',
                      xaxis=dict(
                          rangeselector=dict(
                                  buttons=list([
                                      dict(count=6, label='6m',
                                               step='month', stepmode='backward'),
                                      dict(count=12, label='12m',
                                           step='month', stepmode='backward'),
                                      dict(count=18, label='18m',
                                           step='month', stepmode='backward'),
                                      dict(step='all')])), rangeslider=dict(visible=True), type='date'))

        return fig
    
    def time_doubleexp_plot(self, columns, alpha, beta):
        """
        Построение графика временного ряда с двойным экспоненциальным сглаживанием
        Args:
            columns - отображаемые временные ряды
            alpha - коэффициент экспоненциального сглаживания
            beta - коэффициент экспоненциального сглаживания
        """       
        fig = go.Figure()
    
        def double_exponential_smoothing(series, alpha, beta):
            result = [series[0]]
            for n in range(1, len(series)):
                if n == 1:
                    level, trend = series[0], series[1] - series[0]
                if n >= len(series):
                    value = result[-1]
                else:
                    value = series[n]
                last_level, level = level, alpha*value + (1-alpha)*(level+trend)
                trend = beta*(level-last_level) + (1-beta)*trend
                result.append(level+trend)
        
            return pd.Series(result, index=series.index)

        for prod in columns:
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=self.data[prod], 
                                     name=prod))
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=double_exponential_smoothing(self.data[prod], alpha, beta), 
                                     name=prod+f'_smoothing (a={alpha}, b={beta})'))

        fig.update_layout(yaxis_title="Sales", 
                          xaxis_title="Time",
                          title=f"Sales for {columns} with doubleexp_smoothing",
                          template='plotly_white',
                          xaxis=dict(
                              rangeselector=dict(
                                  buttons=list([
                                      dict(count=6, label='6m',
                                               step='month', stepmode='backward'),
                                      dict(count=12, label='12m',
                                           step='month', stepmode='backward'),
                                      dict(count=18, label='18m',
                                           step='month', stepmode='backward'),
                                      dict(step='all')])), rangeslider=dict(visible=True), type='date'))

        return fig
    
    def outliers_plot(self, value_col, out_col):
        """
        Построение графика временного ряда с отображением "выбросов"
        Args:
            value_col - отображаемый временной ряд
            out_col - признак с найденными выбросами
         """    
        out_ind = []
        out_val = []
        for index, row in self.data.iterrows():
            if row[out_col]==-1:
                out_ind.append(index)
                out_val.append(row[value_col])
                
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, 
                                 y=self.data[value_col], 
                                 name=value_col))
        fig.add_trace(go.Scatter(x=out_ind, 
                                 y=out_val, 
                                 name='outliers', 
                                 mode = "markers", 
                                 marker = dict(color = "red", size = 12)))
        fig.update_layout(height=600, width=1000, title_text=f"{value_col} values with outliers", template='plotly_white')
        return fig
    
    def time_forecast_plot(self, train, actual, predict, model_name):
        """
        Построение графика с тренировочными, тестовымими и предсказанными значениями
        Args:
            train_df - dataframe с тренировочными данными
            test_df - dataframe с тестовыми данными
            train - тренировочные значения
            actual - тестовые значения
            predict - предсказанные значения
            model_name - название модели
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.train_data.index, y=train, name='train'))
        fig.add_trace(go.Scatter(x=self.test_data.index, y=actual, name='test'))
        fig.add_trace(go.Scatter(x=self.test_data.index, y=predict, name='predict'))
        fig.update_layout(height=900, width=900, title_text=f"Predictions {model_name}", template='plotly_white')
        return fig
