import matplotlib.pyplot as plt
import numpy as np



def plot_clients_selection_ratios(load_path: str, save_path: str=None):
    
    # arr = np.load('p1_hasnets/__paper__/gtsrb_hidden_values_visible_backdoor_initially_good.npz')
    arr = np.load(load_path)

    clients_trust_state_history = arr['clients_trust_state_history']
    clients_gamma_history = arr['clients_gamma_history']
    clean_clients_gamma_history = arr['clean_clients_gamma_history']
    nonclean_clients_gamma_history = arr['nonclean_clients_gamma_history']
    clients_selection_ratio = arr['clients_selection_ratio']
    
    processed_clients_selection_ratio = clients_selection_ratio.copy()
    processed_clients_selection_ratio[0] = 0.
    for i in range(1, len(processed_clients_selection_ratio)):
        processed_clients_selection_ratio[i, np.where(processed_clients_selection_ratio[i]==-2)] = processed_clients_selection_ratio[i-1, np.where(processed_clients_selection_ratio[i]==-2)]
        processed_clients_selection_ratio[i, np.where(processed_clients_selection_ratio[i]==-1)] = processed_clients_selection_ratio[i-1, np.where(processed_clients_selection_ratio[i]==-1)]*(i-1)/i
        processed_clients_selection_ratio[i, np.where(processed_clients_selection_ratio[i]==1)] = processed_clients_selection_ratio[i-1, np.where(processed_clients_selection_ratio[i]==1)]*(i-1)/i + 1/i
    # processed_clients_selection_ratio /= len(processed_clients_selection_ratio)

    fig = plt.figure(figsize=(4, 3))

    plt.plot(np.mean(processed_clients_selection_ratio[:, 45:], axis=1), label='Clean Clients')
    plt.fill_between(
        np.arange(len(processed_clients_selection_ratio)), 
        np.min(processed_clients_selection_ratio[:, 45:], axis=1), 
        np.max(processed_clients_selection_ratio[:, 45:], axis=1),
        alpha=0.2
    )

    plt.plot(np.mean(processed_clients_selection_ratio[:, :45], axis=1), label='Backdoored Clients')
    plt.fill_between(
        np.arange(len(processed_clients_selection_ratio)), 
        np.min(processed_clients_selection_ratio[:, :45], axis=1), 
        np.max(processed_clients_selection_ratio[:, :45], axis=1),
        alpha=0.2
    )

    plt.ylabel('Clients\' Selection Ratio')
    plt.xlabel('Training Round: $t$')
    plt.legend(ncols=2, bbox_to_anchor=(1,1.2), edgecolor='white')
    plt.tight_layout()
    
    return fig


def plot_clients_gamma_history(load_path: str, save_path: str=None):
    
    # arr = np.load('p1_hasnets/__paper__/gtsrb_hidden_values_visible_backdoor_initially_good.npz')
    arr = np.load(load_path)

    clients_trust_state_history = arr['clients_trust_state_history']
    clients_gamma_history = arr['clients_gamma_history']
    clean_clients_gamma_history = arr['clean_clients_gamma_history']
    nonclean_clients_gamma_history = arr['nonclean_clients_gamma_history']
    clients_selection_ratio = arr['clients_selection_ratio']
    
    fig = plt.figure(figsize=(4, 3))

    plt.plot(np.mean(clients_gamma_history[:, 45:], axis=1), label='Clean Clients')
    plt.fill_between(
        np.arange(len(clients_gamma_history)), 
        np.min(clients_gamma_history[:, 45:], axis=1), 
        np.max(clients_gamma_history[:, 45:], axis=1),
        alpha=0.2
    )

    plt.plot(np.mean(clients_gamma_history[:, :45], axis=1), label='Backdoored Clients')
    plt.fill_between(
        np.arange(len(clients_gamma_history)), 
        np.min(clients_gamma_history[:, :45], axis=1), 
        np.max(clients_gamma_history[:, :45], axis=1),
        alpha=0.2
    )

    plt.ylabel('Average Trust Index: $\\gamma_i$')
    plt.xlabel('Training Round: $t$')
    plt.legend(ncols=2, bbox_to_anchor=(1,1.2), edgecolor='white')
    plt.tight_layout()
    
    return fig


def plot_clients_trust_history(load_path: str, save_path: str=None):
    
    # arr = np.load('p1_hasnets/__paper__/gtsrb_hidden_values_visible_backdoor_initially_good.npz')
    arr = np.load(load_path)

    clients_trust_state_history = arr['clients_trust_state_history']
    clients_gamma_history = arr['clients_gamma_history']
    clean_clients_gamma_history = arr['clean_clients_gamma_history']
    nonclean_clients_gamma_history = arr['nonclean_clients_gamma_history']
    clients_selection_ratio = arr['clients_selection_ratio']
    
    fig = plt.figure(figsize=(4, 3))

    plt.plot(np.mean(clients_trust_state_history[:, 45:], axis=1), label='Clean Clients')
    plt.fill_between(
        np.arange(len(clients_trust_state_history)), 
        np.min(clients_trust_state_history[:, 45:], axis=1), 
        np.max(clients_trust_state_history[:, 45:], axis=1),
        alpha=0.3
    )

    plt.plot(np.mean(clients_trust_state_history[:, :45], axis=1), label='Backdoored Clients')
    plt.fill_between(
        np.arange(len(clients_trust_state_history)), 
        np.min(clients_trust_state_history[:, :45], axis=1), 
        np.max(clients_trust_state_history[:, :45], axis=1),
        alpha=0.3
    )

    min_ = np.min(clients_trust_state_history)
    max_ = np.max(clients_trust_state_history)
    # plt.fill_between(np.arange(80, 88), min_-10, max_+10, color='green', alpha=0.5, linestyles='dashed')
    plt.vlines([30], min_-10, max_+10, colors='black', linestyles='dashed')
    
    plt.ylim([min_, max_])
    plt.ylabel('Trust State History: $\\phi_i$')
    plt.xlabel('Training Round: $t$')
    plt.legend(ncols=2, bbox_to_anchor=(1,1.2), edgecolor='white')
    plt.tight_layout()
    
    return fig


def plot_clients_gamma_values_over_rounds(load_path: str, save_path: str=None):
    
    # arr = np.load('p1_hasnets/__paper__/gtsrb_hidden_values_visible_backdoor_initially_good.npz')
    arr = np.load(load_path)

    clients_trust_state_history = arr['clients_trust_state_history']
    clients_gamma_history = arr['clients_gamma_history']
    clean_clients_gamma_history = arr['clean_clients_gamma_history']
    nonclean_clients_gamma_history = arr['nonclean_clients_gamma_history']
    clients_selection_ratio = arr['clients_selection_ratio']
    
    processed_clean_clients_gamma_history = clean_clients_gamma_history.copy()
    processed_nonclean_clients_gamma_history = nonclean_clients_gamma_history.copy()
    for i in range(len(processed_clean_clients_gamma_history)):
        if -10 in processed_clean_clients_gamma_history[i]:
            processed_clean_clients_gamma_history[i, np.where(processed_clean_clients_gamma_history[i]==-10)] = np.mean(processed_clean_clients_gamma_history[i, np.where(processed_clean_clients_gamma_history[i]!=-10)])
        if -10 in processed_nonclean_clients_gamma_history[i]:
            processed_nonclean_clients_gamma_history[i, np.where(processed_nonclean_clients_gamma_history[i]==-10)] = np.mean(processed_nonclean_clients_gamma_history[i, np.where(processed_nonclean_clients_gamma_history[i]!=-10)])
        # if np.max(processed_clean_clients_gamma_history[i]) == np.min(processed_clean_clients_gamma_history[i]):
        #     processed_clean_clients_gamma_history[i,0] -= 1e-4 * processed_clean_clients_gamma_history[i,0]
        # if np.max(processed_nonclean_clients_gamma_history[i]) == np.min(processed_nonclean_clients_gamma_history[i]):
        #     processed_nonclean_clients_gamma_history[i,0] -= 1e-4 * processed_nonclean_clients_gamma_history[i,0]
        

    fig = plt.figure(figsize=(4, 3))
    
    plt.plot(np.mean(processed_clean_clients_gamma_history, axis=1), label='Clean Clients')
    plt.fill_between(
        np.arange(len(processed_clean_clients_gamma_history)), 
        np.min(processed_clean_clients_gamma_history, axis=1), 
        np.max(processed_clean_clients_gamma_history, axis=1),
        alpha=0.2
    )

    plt.plot(np.mean(processed_nonclean_clients_gamma_history, axis=1), label='Backdoored Clients')
    plt.fill_between(
        np.arange(len(processed_clean_clients_gamma_history)), 
        np.min(processed_nonclean_clients_gamma_history, axis=1), 
        np.max(processed_nonclean_clients_gamma_history, axis=1),
        alpha=0.2
    )
    
    min_ = min(np.min(processed_clean_clients_gamma_history), np.min(processed_nonclean_clients_gamma_history))
    max_ = max(np.max(processed_clean_clients_gamma_history), np.max(processed_nonclean_clients_gamma_history))
    # plt.fill_between(np.arange(80, 88), min_-10, max_+10, color='green', alpha=0.5, linestyles='dashed')
    plt.vlines([30], min_-10, max_+10, colors='black', linestyles='dashed')
    
    plt.ylim([min_, max_])
    plt.ylabel('Average Trust Index')
    plt.xlabel('Training Round: $t$')
    plt.legend(ncols=2, bbox_to_anchor=(1,1.2), edgecolor='white')
    plt.tight_layout()
    
    return fig

