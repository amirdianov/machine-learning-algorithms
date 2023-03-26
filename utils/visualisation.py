import plotly.graph_objects as go


class Visualisation:
    @staticmethod
    def target_func(backup):
        x, y = [], []
        fig = go.Figure()
        for pair in backup:
            x.append(pair[0])
            y.append(pair[1])
        trace = go.Scatter(
            x=x, y=y, mode="lines", name='Graphic'
        )
        fig.add_trace(trace)
        fig.update_layout(
            title=f"График зависимости target value от номера итерации"
        )
        title = 'Target_value'
        fig.write_html(f"{title}.html")
        fig.show()
