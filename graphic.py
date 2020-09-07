import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



dd1 = pd.read_csv('d3_result/m4_fusej.txt', header=None)
dd2 = pd.read_csv('d3_result/m4_fuser.txt', header=None)
dd3 = pd.read_csv('d3_result/m4_pfm.txt', header=None)
dd4 = pd.read_csv('d3_result/m4_nfpfm.txt', header=None)
dd5 = pd.read_csv('d3_result/rgb.txt', header=None)
d1i = pd.read_csv('d3_result/ir.txt', header=None)

dd6 = pd.read_csv('d2_result/m4_fusej.txt', header=None)
dd7 = pd.read_csv('d2_result/m4_fuser.txt', header=None)
dd8 = pd.read_csv('d2_result/m4_pfm.txt', header=None)
dd9 = pd.read_csv('d2_result/m4_nfpfm.txt', header=None)
dd10 = pd.read_csv('d2_result/rgb.txt', header=None)
d2i = pd.read_csv('d2_result/ir.txt', header=None)

dd11 = pd.read_csv('d1_result/m4_fusej.txt', header=None)
dd12 = pd.read_csv('d1_result/m4_fuser.txt', header=None)
dd13 = pd.read_csv('d1_result/m4_pfm.txt', header=None)
dd14 = pd.read_csv('d1_result/m4_nfpfm.txt', header=None)
dd15 = pd.read_csv('d1_result/rgb.txt', header=None)
d3i = pd.read_csv('d1_result/ir.txt', header=None)

d1r = pd.read_csv('d1_result/r.txt',header=None)
d1g = pd.read_csv('d1_result/g.txt',header=None)
d1b = pd.read_csv('d1_result/b.txt',header=None)

d2r = pd.read_csv('d2_result/r.txt',header=None)
d2g = pd.read_csv('d2_result/g.txt',header=None)
d2b = pd.read_csv('d2_result/b.txt',header=None)

d3r = pd.read_csv('d3_result/r.txt',header=None)
d3g = pd.read_csv('d3_result/g.txt',header=None)
d3b = pd.read_csv('d3_result/b.txt',header=None)



def create_graph_data(dataframe):

    data = [row.split(' ') for row in dataframe[0]]
    data = data[1:-1]
    data = [(float(row[8][:-7]), float(row[13])) for row in data]
    accuracy = [tup[0] for tup in data]
    loss = [tup[1] for tup in data]
    return accuracy, loss

def create_graph_data_two(dataframe):

    data = [row.split(' ') for row in dataframe[0]]
    data = data[1:-1]
    data = [(float(row[7][:-7]), float(row[12])) for row in data]
    accuracy = [tup[0] for tup in data]
    loss = [tup[1] for tup in data]
    return accuracy, loss


a1, l1 = create_graph_data(dd1)
a2, l2 = create_graph_data(dd2)
a3, l3 = create_graph_data(dd3)
a4, l4 = create_graph_data(dd4)
#a5, l5 = create_graph_data_two(dd5)

a6, l6 = create_graph_data(dd6)
a7, l7 = create_graph_data(dd7)
a8, l8 = create_graph_data(dd8)
a9, l9 = create_graph_data(dd9)
#a10, l10 = create_graph_data(dd10)

a11, l11 = create_graph_data(dd11)
a12, l12 = create_graph_data(dd12)
a13, l13 = create_graph_data(dd13)
a14, l14 = create_graph_data(dd14)
#a15, l15 = create_graph_data(dd15)

#air1, lir1 = create_graph_data(d1i)
#air2, lir2 = create_graph_data(d2i)
#air3, lir3 = create_graph_data(d3i)

ar1, lr1 = create_graph_data(d1r)
ar2, lr2 = create_graph_data(d2r)
ar3, lr3 = create_graph_data(d3r)

ag1, lg1 = create_graph_data(d1g)
ag2, lg2 = create_graph_data(d2g)
ag3, lg3 = create_graph_data(d3g)

ab1, lb1 = create_graph_data(d1b)
ab2, lb2 = create_graph_data(d2b)
ab3, lb3 = create_graph_data(d3b)


fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(6,10))

ax1.set_title('Dataset 1 - Training Loss', fontsize=14)
ax2.set_title('Dataset 2 - Training Loss', fontsize=14)
ax3.set_title('Dataset 3 - Training Loss', fontsize=14)

ax1.set_xlabel('Epoch', fontsize=8)
ax2.set_xlabel('Epoch', fontsize=8)
ax3.set_xlabel('Epoch', fontsize=8)

ax1.set_ylabel('Training Loss', fontsize=8)
ax2.set_ylabel('Training Loss', fontsize=8)
ax3.set_ylabel('Training Loss', fontsize=8)

#ax1.set_ylim(bottom=85, top=100)
#ax2.set_ylim(bottom=85, top=100)
#ax3.set_ylim(bottom=85, top=100)

ax1.plot(l11, label='fusej')
ax1.plot(l12, label='fuser')
ax1.plot(l13, label='pfm')
ax1.plot(l14, label='nfpfm')
ax1.plot(lr1, label='r')
ax1.plot(lg1, label='g')
ax1.plot(lb1, label='b')
#ax1.plot(a15, label='rgb')
#ax1.plot(air1, label='ir')

ax2.plot(l6, label='fusej')
ax2.plot(l7, label='fuser')
ax2.plot(l8, label='pfm')
ax2.plot(l9, label='nfpfm')
ax2.plot(lr2, label='r')
ax2.plot(lg2, label='g')
ax2.plot(lb2, label='b')
#ax2.plot(a10, label='rgb')
#ax2.plot(air2, label='ir')

ax3.plot(l1, label='fusej')
ax3.plot(l2, label='fuser')
ax3.plot(l3, label='pfm')
ax3.plot(l4, label='nfpfm')
ax3.plot(lr3, label='r')
ax3.plot(lg3, label='g')
ax3.plot(lb3, label='b')
#ax3.plot(a5, label='rgb')
#ax3.plot(air3, label='ir')

ax1.grid(linestyle=':', linewidth=1)
ax2.grid(linestyle=':', linewidth=1)
ax3.grid(linestyle=':', linewidth=1)

plt.tight_layout()
ax1.legend()
ax2.legend()
ax3.legend()

plt.savefig('training_loss_m4.pdf',bbox_inches='tight', transparent=True)



