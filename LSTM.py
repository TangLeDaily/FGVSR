import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# Define some constants



class ConvLSTMCell(nn.Module):
    """
    init: in_c, out_c
    input: tensor(x): b, in_c, h, w ; tensor(pre_x): b, out_c, h, w ; tensor(pre_c): b, out_c, h, w
    out: tensor(y & pre_h): b, out_c, h, w ; tensor(pre_c): b, out_c, h, w
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3//2)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state
        prev_hidden = prev_hidden.cuda()
        prev_cell = prev_cell.cuda()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        return hidden, cell


def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    b, c, h, w = 8, 64, 128, 128
    d = 64           # hidden state size
    lr = 1e-1       # learning rate
    T = 6           # sequence length
    max_epoch = 2000  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(c, d)
    print(repr(model))

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, h, w))
    y = Variable(torch.randn(T, b, d, h, w))

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()
        for t in range(0, T):
            state = model(x[t], state)

            loss += loss_fn(state[0], y[t])

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))

class DSLSTM(nn.Module):
    """
    init: in_c, out_c
    input: tensor(x): b, in_c, h, w ; tensor(pre_x): b, out_c, h, w ; tensor(pre_c): b, out_c, h, w
    out: tensor(y & pre_h): b, out_c, h, w ; tensor(pre_c): b, out_c, h, w
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size = 64, hidden_size = 64):
        super(DSLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.h1_conv = nn.Conv2d(hidden_size, 1, 3, padding=3//2)
        self.h2_conv = nn.Conv2d(hidden_size, 1, 3, padding=3//2)
        self.h3_conv = nn.Conv2d(hidden_size, 1, 3, padding=3 // 2)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)

    def forward(self, input_, np_feat, h_suv, np_feat_c, c_suv):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        # generate empty prev_state, if None is provided
        if c_suv:
            prev_cell = np_feat_c
        else:
            prev_cell = Variable(torch.zeros(state_size))
        prev_hidden = []
        for i in range(3):
            if h_suv[i]:
                prev_hidden.append(np_feat[i])
            else:
                prev_hidden.append(Variable(torch.zeros(state_size)))

        prev_hidden = prev_hidden.cuda()
        prev_cell = prev_cell.cuda()
        h1 = f.relu(self.h1_conv(prev_hidden[0]))
        h2 = f.relu(self.h2_conv(prev_hidden[1]))
        h3 = f.relu(self.h3_conv(prev_hidden[3]))
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, h1, h2, h3), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        return hidden, cell

if __name__ == '__main__':
    _main()