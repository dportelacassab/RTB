import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

class Visualization_:

    def __init__(self,data_prep):
        self.data_prep = data_prep        # self.d = data_.values
        self.row_ = self.data_prep.shape[0]
        self.col_ = self.data_prep.shape[1]
        self.variables = list(self.data_prep.columns)

    def countPlot(self):
        plt.figure(figsize=(30, 30))  # add this line if you want to set the figure size!
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=1.3, wspace=0.8, hspace=0.4)
        for g in range(0, len(self.variables) ):
            plt.subplot(3, 2, g + 1)  # #rows , #cols, plot order.
            sns.countplot(x=self.variables[g], data=self.data_prep);
            plt.title('Countplot for ' + self.variables[g]);
            plt.xticks(fontsize=7, rotation=45, ha="right")
        plt.savefig("countplot.pdf", bbox_inches='tight')

    def map(self):
        ge3 = self.data_prep["geo_region"].astype("category").value_counts() / self.row_
        df = pd.DataFrame(ge3)
        df['geo'] = df.index

        fig = px.choropleth(df,
                            locations='geo',
                            locationmode="USA-states",
                            scope="usa",
                            color='geo_region',
                            color_continuous_scale="Viridis_r",
                            )
        fig.show()

        data_crosstab = pd.crosstab(self.data_prep['geo_region'], self.data_prep['click'], margins=False);
        print(data_crosstab)
        da = {'geo': data_crosstab.iloc[:, 1].index, 'clicks': data_crosstab.iloc[:, 1].values};df = pd.DataFrame(da)
        fig = px.choropleth(df,
                            locations='geo',
                            locationmode="USA-states",
                            scope="usa",
                            color='clicks',
                            color_continuous_scale="Viridis_r",
                            )
        fig.show()
