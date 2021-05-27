import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import statsmodels.formula.api as smf


def exponential(t: int, r: float, x0: float) -> float:
    return x0 * np.power((1 + r), t)


class ExponentialFit:
    def __init__(self, df: pd.DataFrame, max_days: int = 30, min_cases: int = 20,
                 cases_col: str = 'cumulative_cases', country_col: str = 'country',
                 fix_x0: bool = False, optimization_kwargs: dict = None, log_norm: bool = False):

        optimization_kwargs = optimization_kwargs if optimization_kwargs is not None else {}
        try:
            assert df[country_col].nunique() == 1
        except AssertionError:
            raise ValueError(f'Expected DataFrame with data from single country,'
                             f' but {df[country_col].unique()} were found')
        self.log_norm = log_norm
        self.country = df[country_col].unique()[0]
        self.country_col = country_col
        y_data = (df.loc[lambda dd: dd[cases_col] >= min_cases])[cases_col].values
        self.min_cases = min_cases
        self.y_data = y_data if max_days >= len(y_data) else y_data[:max_days]
        self.x_data = range(len(self.y_data))
        if log_norm:
            y = np.log(self.y_data)
            X = np.array(self.x_data).reshape(-1, 1)
            lm = LinearRegression()
            lm.fit(X, y)
            self.x_0 = lm.intercept_
            self.r = lm.coef_[0]
            self.score = lm.score(X, y)
            log_preds = lm.predict(X)
            self.MSE = np.square(y - log_preds).mean()
            self.y_pred = np.exp(log_preds)

        else:
            if fix_x0:
                self.x_0 = y_data.min()
                self.f = lambda t, r: self.x_0 * (1 + r) ** t
            else:
                self.f = exponential
                self.x_0 = None  # Define after fit

            popt, pcov = curve_fit(self.f, self.x_data, self.y_data, maxfev=10e6,
                                   bounds=(0, np.inf), **optimization_kwargs)
            self.r = popt[0]
            self.x_0 = self.x_0 if fix_x0 else popt[1]
            self.y_pred = self.f(self.x_data, *popt)
            self.MSE = np.square(self.y_data - self.y_pred).mean()

        self.predictions_df = pd.DataFrame({self.country_col: self.country, 'day': self.x_data,
                                            'true_cases': self.y_data,
                                            'predicted_cases': self.y_pred})

    def plot_fit(self, log_scale: bool = False, **kwargs):
        if self.log_norm:
            exp_lab = fr'$log(f(t)) = x_0 * rt; \ '
        else:
            exp_lab = fr'$f(t) = x_0 * (1 + r) ^ t; \ '
        fig, ax = plt.subplots(dpi=120, **kwargs)
        plt.plot(self.x_data, self.y_data, 'b-', label='data')
        plt.plot(self.x_data, self.y_pred, 'g--',
                 label=exp_lab + fr'x_0 = {round(self.x_0, 2)}, r = {round(self.r * 100, 2)}\%$')
        plt.xlabel(f'Days with more than {self.min_cases} cases')
        plt.ylabel('COVID-19 cumulative cases')
        plt.title(f'Fit for {self.country}')
        if log_scale:
            plt.yscale('log')
        plt.legend()

        return fig, ax


class CountryFittedComparison:
    def __init__(self, df, max_days: int = 30, min_cases: int = 20,
                 cases_col: str = 'cumulative_cases',
                 country_col: str = 'country', min_days: int = 10, fix_x0: bool = False,
                 log_norm: bool = False):
        self.country_col = country_col
        self.min_cases = min_cases
        self.cases_col = cases_col
        self.country_fits = {}
        for country, country_df in df.groupby(country_col):
            if country_df.loc[lambda dd: dd[cases_col] >= min_cases].shape[0] >= min_days:
                self.country_fits[country] = ExponentialFit(country_df, max_days=max_days,
                                                            min_cases=min_cases,
                                                            log_norm=log_norm, cases_col=cases_col,
                                                            country_col=country_col, fix_x0=fix_x0)

        self.country_pars_df = pd.DataFrame({self.country_col: list(self.country_fits.keys()),
                                             'r': [c.r for c in self.country_fits.values()],
                                             'x_0': [c.x_0 for c in self.country_fits.values()],
                                             'MSE': [c.MSE for c in self.country_fits.values()],
                                             'max_cases': [c.y_data.max() for c in
                                                           self.country_fits.values()],
                                             'total_days': [len(c.y_data) for c in
                                                            self.country_fits.values()]})

        self.country_preds_df = pd.concat([c.predictions_df for c in self.country_fits.values()])

    def plot_countries_comparison(self, min_days: int = 25, log_scale: bool = False):
        p = (self.country_preds_df
             .loc[lambda dd: dd[self.country_col].isin(
             self.country_pars_df.loc[lambda df: df.total_days >= min_days][self.country_col]
             )]
             .merge(self.country_pars_df)
             .assign(facet_str=lambda dd: dd[self.country_col] + ': r = ' +
                                          (dd.r * 100).round(2).astype(str) + '%')
             .pipe(lambda dd: p9.ggplot(dd)
                              + p9.aes('day', 'true_cases')
                              + p9.geom_line()
                              + p9.geom_line(p9.aes(y='predicted_cases'), linetype='dashed',
                                             color='darkred')
                              + p9.facet_wrap('facet_str', scales='free_y', ncol=3)
                              + p9.theme_bw()
                              + p9.theme(figure_size=(12, 1.5 * ((dd[self.country_col].nunique()
                                                                  - 1) // 3 + 1)),
                                         subplots_adjust={'wspace': 0.2})
                              + p9.labs(x=f'Days after {self.min_cases} cases',
                                        y='Cumulative cases'))
             )

        if log_scale:
            p += p9.scale_y_log10()

        return p
