import plotly.graph_objects as go


class Visualisation:
    @staticmethod
    def mixin(data, title):
        x, y = [], []
        fig = go.Figure()
        for pair in data:
            if pair[0] not in x:
                x.append(pair[0])
                y.append(pair[1])
        trace = go.Scatter(
            x=x, y=y, mode="lines", name='Graphic'
        )
        fig.add_trace(trace)
        fig.update_layout(
            title=f"График зависимости {title} от номера итерации"
        )
        # fig.write_html(f"{title}.html")
        fig.show()

    @staticmethod
    def visualisation(backup):
        Visualisation.mixin(backup['target_value_func'], title='target_value_func')
        Visualisation.mixin(backup['accuracy_train'], title='accuracy_train')
        Visualisation.mixin(backup['accuracy_valid'], title='accuracy_valid')
