import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_metrics(path, history, use_mas):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    if use_mas and len(history['train']['epoch']) == len(history['train']['mas_dis']):
        plt.plot(history['train']['epoch'], history['train']['mas_dis'],
                 color='g', label='mas dis')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['sdr'],
             color='r', label='SDR')
    plt.plot(history['val']['epoch'], history['val']['sir'],
             color='g', label='SIR')
    plt.plot(history['val']['epoch'], history['val']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')
