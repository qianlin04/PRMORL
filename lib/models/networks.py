from __future__ import absolute_import, division, print_function
from numpy.core.fromnumeric import reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utilities import common_utils
from torch.autograd import Variable

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

############################################################################################
# Function to initialize weights
############################################################################################
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class MO_DDQN(nn.Module):
    '''
        Multi-objective version of DDQN
    '''

    def __init__(self, args):
        super(MO_DDQN, self).__init__()


        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        
                   
        
        self.affine_in = nn.Linear(self.state_size + self.reward_size,self.hidden_size)
        self.affine = nn.Linear(self.hidden_size,self.hidden_size)
        self.affine_hid = common_utils.get_clones(self.affine, self._layer_N)
        del self.affine
        self.affine_out =  nn.Linear(self.hidden_size, self.action_size * self.reward_size)
        

        self.affine_in.apply(init_weights)
        self.affine_hid.apply(init_weights)
        self.affine_out.apply(init_weights)
        

    def forward(self, state, preference):
        
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.affine_in(x))
        for i in range(self._layer_N):
            x = F.relu(self.affine_hid[i](x))
        q = self.affine_out(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)
        

        return q


class Actor(nn.Module):
    def __init__(self, args, is_gaussian=False):
        super(Actor,self).__init__()
        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        self._layer_N = args.layer_N_actor
        self.hidden_size = args.hidden_size
        self.is_gaussian = is_gaussian
        
        self.affine_in = nn.Linear(self.state_size + self.reward_size,self.hidden_size)
        self.affine = nn.Linear(self.hidden_size,self.hidden_size)
        self.affine_hid = common_utils.get_clones(self.affine, self._layer_N)
        del self.affine
        self.affine_out =  nn.Linear(self.hidden_size, self.action_size*2 if self.is_gaussian else self.action_size)
        
        self.affine_in.apply(init_weights)
        self.affine_hid.apply(init_weights)
        self.affine_out.apply(init_weights)
        
        self.max_action = torch.FloatTensor(args.max_action).to(self.args.device)
        
    def forward(self, state, preference):
        
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.affine_in(x))
        for i in range(self._layer_N):
            x = F.relu(self.affine_hid[i](x))
        x = self.affine_out(x)
        if not self.is_gaussian:
            x = torch.tanh(x)
            return self.max_action*x
        else:
            mean, log_std = x[:, :self.action_size], x[:, self.action_size:]
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            return mean, log_std
    
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic,self).__init__()
        
        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        self._layer_N = args.layer_N_critic
        self.hidden_size = args.hidden_size
        
        
        # Q1 architecture
        self.affine_in_1 = nn.Linear(self.state_size + self.reward_size + self.action_size,self.hidden_size)
        self.affine_1 = nn.Linear(self.hidden_size,self.hidden_size)
        self.affine_hid_1 = common_utils.get_clones(self.affine_1, self._layer_N)
        del self.affine_1
        self.affine_out_1 =  nn.Linear(self.hidden_size, self.reward_size)
        
        # Q2 architecture
        self.affine_in_2 = nn.Linear(self.state_size + self.reward_size + self.action_size,self.hidden_size)
        self.affine_2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.affine_hid_2 = common_utils.get_clones(self.affine_2, self._layer_N)
        del self.affine_2
        self.affine_out_2 =  nn.Linear(self.hidden_size, self.reward_size)
        
        self.affine_in_1.apply(init_weights)
        self.affine_hid_1.apply(init_weights)
        self.affine_out_1.apply(init_weights)
        
        self.affine_in_2.apply(init_weights)
        self.affine_hid_2.apply(init_weights)
        self.affine_out_2.apply(init_weights)
        
        self.max_action = torch.FloatTensor(args.max_action).to(self.args.device)
        
    def forward(self, state, preference, action):
        
        x = torch.cat((state, preference, action), dim=1)
        x = x.view(x.size(0), -1)
        
        x1 = F.mish(self.affine_in_1(x))
        for i in range(self._layer_N):
            x1 = F.mish(self.affine_hid_1[i](x1))
        q1 = self.affine_out_1(x1)
        
        x2 = F.mish(self.affine_in_2(x))
        for i in range(self._layer_N):
            x2 = F.mish(self.affine_hid_2[i](x2))
        q2 = self.affine_out_2(x2)
        return q1,q2
    
    
    def Q1(self,state,preference,action):
        
        x = torch.cat((state, preference, action), dim=1)
        x = x.view(x.size(0), -1) 
        
        x1 = F.mish(self.affine_in_1(x))
        for i in range(self._layer_N):
            x1 = F.mish(self.affine_hid_1[i](x1))
        q1 = self.affine_out_1(x1)
        return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, args, reward_size):
        super(VAE, self).__init__()  
        state_dim = args.obs_shape
        action_dim = args.action_shape
        latent_dim = args.action_shape * 2
        max_action = args.max_action[0]
        device = args.device
        self.e1 = nn.Linear(state_dim + reward_size + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + reward_size + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, pref, action):
        z = F.relu(self.e1(torch.cat([state, pref, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, pref, z)

        return u, mean, std


    def decode(self, state, pref, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, pref, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
class BCQ_Actor(nn.Module):
    def __init__(self, args):
        super(BCQ_Actor, self).__init__()
        state_dim = args.obs_shape
        action_dim = args.action_shape
        max_action = args.max_action[0]
        reward_size = args.reward_size

        self.l1 = nn.Linear(state_dim + reward_size + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.phi = args.bcq_max_phi


    def forward(self, state, pref, action, w_bc):
        a = F.relu(self.l1(torch.cat([state, pref, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * (1-w_bc) * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class EnvelopeLinearCQN_default(nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, args ):
        super(EnvelopeLinearCQN_default, self).__init__()


        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        
        
        self.affine1 = nn.Linear(self.state_size + self.reward_size,
                                    (self.state_size + self.reward_size) * 16)
        self.affine2 = nn.Linear((self.state_size + self.reward_size) * 16,
                                  (self.state_size + self.reward_size) * 32)
        self.affine3 = nn.Linear((self.state_size + self.reward_size) * 32,
                                  (self.state_size + self.reward_size) * 64)
        self.affine4 = nn.Linear((self.state_size + self.reward_size) * 64,
                                  (self.state_size + self.reward_size) * 32)
        self.affine5 = nn.Linear((self.state_size + self.reward_size) * 32,
                                  self.action_size * self.reward_size)
               
        
    def H(self, Q, w, s_num, w_num):
        
        use_cuda = self.args.cuda
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        mask = BoolTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def H_(self, Q, w, s_num, w_num):
        
        use_cuda = self.args.cuda
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
        
        reQ = Q.view(-1, self.reward_size)

        # extend preference batch
        w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions
        prod = prod.view(-1, self.action_size)
        inds = prod.max(1)[1]
        mask = BoolTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def forward(self, state, preference, w_num=1):
        
        s_num = int(preference.size(0) / w_num)
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)

               
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        q = self.affine5(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)   
        

        hq = self.H_(q.detach().view(-1, self.reward_size), preference, s_num, w_num)     

        return q