import matplotlib.pyplot as plt

def plot_time_series(time_series_dict, separate_ax=True):
    """
    The function takes input like:

    time_series_dict = {
    'Hospitalise Rate': {
        'Observed': observed_hospitalise_rate,
        'Simulated': simulated_hospitalise_rate,
        'Errors': error_hospitalise_rate
        },
    'Recover Rate': {
        'Observed': observed_recover_rate,
        'Simulated': simulated_recover_rate,
        'Errors': error_recover_rate
        }
    }
    
    """

    max_value_length = 1
    for _, tss in time_series_dict.items():
        for _, ts in tss.items():
            if len(ts) > max_value_length:
                max_value_length = len(ts)
    # print(max_value_length)
    ax_width = (max(max_value_length, 20) // 20 * 8)
    ax_height = int(ax_width * 0.8)

    if separate_ax:
        
        n_of_vars = len(time_series_dict)
        
        figsize_x = min(50, ax_width * n_of_vars)
        figsize_y = min(40, ax_height)

        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi=100)

        index = 1

        for name, time_series in time_series_dict.items():
            ax = fig.add_subplot(1, n_of_vars, index)
            for nm, ts in time_series.items():
                line, = plt.plot(ts, label=nm, linestyle='--', marker='o')
                x, y = line.get_data()
                for i, j in zip(x, y):
                    ax.annotate(round(j), xy=(i,j))
            ax.set_title(name)
            ax.legend()
            ax.set_xlabel('Time unit')
            ax.set_ylabel('Quantity / Time unit')
            index+=1
    else:
        
        fig = plt.figure(figsize=(min(50, ax_width), min(40, ax_height)))
        ax = fig.add_subplot(1, 1, 1)
        for name, time_series in time_series_dict.items():
            for nm, ts in time_series.items():
                line, = plt.plot(ts, label=nm +' '+name, linestyle='--', marker='o')
                x, y = line.get_data()
                for i, j in zip(x, y):
                    ax.annotate(round(j), xy=(i,j))
        ax.legend()
        ax.set_xlabel('Time unit')
        ax.set_ylabel('Quantity / Time unit')

    plt.show()


if __name__ == "__main__":
    a = [106.49738145, 107.55297435, 118.88731299, 128.80812551, 149.87163052, 151.84484521, 184.13534706]
    plot_time_series({'a': {'aa':a}})
