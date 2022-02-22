import matplotlib.pyplot as plt
import numpy as np
import os

def plot_individual_areas(session, data_class, VAR_results, session_info, save_path=None, start_time=None, end_time=None, return_criticalities=False):
    # SET UP AREAS AND START/END INDICES
    if data_class == 'propofolPuffTone':
        area_colors_all = [('CPB', 'lightsteelblue'), ('7b', 'slategray'), ('FEF', 'skyblue'), ('vlPFC', 'C0')]
    else:
        area_colors_all = [('CPB', 'lightsteelblue'), ('7b', 'slategray'), ('FEF', 'skyblue'), ('vlPFC', 'C0'), ('Thal', 'midnightblue')]

    area_colors = []
    for area, c in area_colors_all:
        if area in VAR_results.keys():
            area_colors.append((area, c))
    
    # PLOT ALL INDIVIDUAL AREAS
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    min_val = np.Inf
    max_val = -np.Inf
    individual_criticalities = {}
    for area, c in area_colors:
        if start_time is None:
            start_ind = 0
        else:
            start_ind = np.abs(VAR_results[area].start_time - start_time).argmin()
        if end_time is None:
            end_ind = len(VAR_results[area])
        else:
            end_ind = np.abs(VAR_results[area].start_time - end_time).argmin()

        start_times = VAR_results[area].start_time/60
        criticality_inds = VAR_results[area].criticality_inds.apply(lambda x: x.mean())
        start_times = start_times[start_ind:end_ind]
        criticality_inds = criticality_inds[start_ind:end_ind]
        individual_criticalities[area] = criticality_inds
        ax.plot(start_times, criticality_inds, label=area, c=c)

        if criticality_inds.min() < min_val:
            min_val = criticality_inds.min()
        if criticality_inds.max() > max_val:
            max_val = criticality_inds.max()


    if data_class == 'propofolPuffTone':
        ax.fill_between(np.arange(session_info['drugStart'][0], session_info['drugEnd'][0])/60, 
                                        min_val, max_val, color='plum', alpha=0.2, label=f"drug infusion 1 - dose = {session_info['drugDose'][0]}")
        ax.fill_between(np.arange(session_info['drugStart'][1], session_info['drugEnd'][1])/60, 
                            min_val, max_val, color='darkorchid', alpha=0.2, label=f"drug infusion 2 - dose = {session_info['drugDose'][1]}")
        plt.axvline(session_info['eyesClose'][-1]/60 if isinstance(session_info['eyesClose'], np.ndarray) else session_info['eyesClose']/60, linestyle='--', c='red', label="loss of consciousness")
        plt.axvline(session_info['eyesOpen'][-1]/60 if isinstance(session_info['eyesOpen'], np.ndarray) else session_info['eyesOpen']/60, linestyle='--', c='green', label="return of consciousness")
    
    elif data_class == 'ketamine':
        colors = ['plum', 'darkorchid']
        for i in range(len(session_info['drugStart'])):
            ax.fill_between(np.arange(session_info['drugStart'][i], session_info['drugEnd'][i])/60, 
                                        min_val, max_val, color=colors[i], alpha=0.2, label=f"drug infusion 1 ({session_info['drug'][i]})")
    
    ax.legend(fontsize=14)        
    # fig.text(0.52, -0.02, 'Time (min)', ha='center', fontsize=16)
    ax.set_ylabel('Mean Criticality Index', fontsize=16)
    ax.set_xlabel('Time in Session (min)', fontsize=16)
    ax.tick_params(labelsize=13)
    # plt.suptitle(f"Mean Criticality Index of VAR Transition Matrix - Monkey {1 if 'Mary' in session else 2}\nWindow = {windows} s", fontsize=18)
    plt.suptitle(f"Mean Criticality Index of VAR Transition Matrix - Monkey {1 if 'Mary' in session else 2}", fontsize=18)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

    if return_criticalities:
        return individual_criticalities

def plot_multipopulation(session, data_class, VAR_results, session_info, electrode_info, save_path=None, start_time=None, end_time=None, return_criticalities=False):
    # SET UP AREAS AND START/END INDICES
    if data_class == 'propofolPuffTone':
        area_colors_all = [('CPB', 'lightsteelblue'), ('7b', 'slategray'), ('FEF', 'skyblue'), ('vlPFC', 'C0')]
    else:
        area_colors_all = [('CPB', 'lightsteelblue'), ('7b', 'slategray'), ('FEF', 'skyblue'), ('vlPFC', 'C0'), ('Thal', 'midnightblue')]

    area_colors = []
    for area, c in area_colors_all:
        if area in VAR_results.keys():
            area_colors.append((area, c))
    
    if start_time is None:
        start_ind = 0
    else:
        start_ind = np.abs(VAR_results['all'].start_time - start_time).argmin()
    if end_time is None:
        end_ind = len(VAR_results['all'])
    else:
        end_ind = np.abs(VAR_results['all'].start_time - end_time).argmin()

    # PLOT MULTIPOPULATION
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()

    min_val = np.Inf
    max_val = -np.Inf
    multipop_criticalities = {}
    for area, c in area_colors:
        start_times = VAR_results['all'].start_time/60
        unit_indices = np.where(electrode_info['area'] == area)[0]

        criticality_inds = np.zeros(len(VAR_results['all']))
        for i in range(len(criticality_inds)):
            e, _ = np.linalg.eig(VAR_results['all'].A_mat.iloc[i][unit_indices][:, unit_indices])
            criticality_inds[i] = np.abs(e).mean()
        
        start_times = start_times[start_ind:end_ind]
        criticality_inds = criticality_inds[start_ind:end_ind]
        multipop_criticalities[area] = criticality_inds
        ax.plot(start_times, criticality_inds, label=area, c=c)

        if criticality_inds.min() < min_val:
            min_val = criticality_inds.min()
        if criticality_inds.max() > max_val:
            max_val = criticality_inds.max()
    
    if data_class == 'propofolPuffTone':
        ax.fill_between(np.arange(session_info['drugStart'][0], session_info['drugEnd'][0])/60, 
                                        min_val, max_val, color='plum', alpha=0.2, label=f"drug infusion 1 - dose = {session_info['drugDose'][0]}")
        ax.fill_between(np.arange(session_info['drugStart'][1], session_info['drugEnd'][1])/60, 
                                min_val, max_val, color='darkorchid', alpha=0.2, label=f"drug infusion 2 - dose = {session_info['drugDose'][1]}")

        plt.axvline(session_info['eyesClose'][-1]/60 if isinstance(session_info['eyesClose'], np.ndarray) else session_info['eyesClose']/60, linestyle='--', c='red', label="loss of consciousness")
        plt.axvline(session_info['eyesOpen'][-1]/60 if isinstance(session_info['eyesOpen'], np.ndarray) else session_info['eyesOpen']/60, linestyle='--', c='green', label="return of consciousness")
    ax.legend(fontsize=14)        
    fig.text(0.52, -0.02, 'Time (min)', ha='center', fontsize=16)
    ax.set_ylabel('Mean Criticality Index', fontsize=16)
    ax.set_xlabel('Time in Session (min)', fontsize=16)
    ax.tick_params(labelsize=13)
    # plt.suptitle(f"Mean Criticality Index of VAR Transition Matrix (Multipop) - Monkey {1 if 'Mary' in session else 2}\nWindow = {window} s", fontsize=18)
    plt.suptitle(f"Mean Criticality Index of VAR Transition Matrix (Multipop) - Monkey {1 if 'Mary' in session else 2}", fontsize=18)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

    if return_criticalities:
        return multipop_criticalities