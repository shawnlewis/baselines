from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os
import json
import copy
import random

def traj_segment_generator(pi, env, horizon, stochastic, action_bias=0.4, action_repeat=0, action_repeat_rand=False, warmup_frames=0):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    repeat_left = 0

    warmup_left = warmup_frames

    while True:
        prevac = ac

        if warmup_left:
            if warmup_left == warmup_frames:
                ac = env.action_space.sample().round() - .2
            _, _, new, _ = env.step(ac)
            warmup_left -= 1
            if new:
                env.reset()
            continue

        if repeat_left:
            repeat_left -= 1
        else:
            ac, vpred = pi.act(stochastic, ob)
            if action_repeat_rand:
                repeat_left = random.randrange(action_repeat)
            else:
                repeat_left = action_repeat
                full_step_rew = 0

        if repeat_left == action_repeat:
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

        ob, rew, new, _ = env.step(action_bias + ac)
        full_step_rew += rew

        if repeat_left == action_repeat:
            rews[i] = full_step_rew

            cur_ep_ret += full_step_rew
            cur_ep_len += 1

            full_step_rew = 0

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            warmup_left = warmup_frames
            repeat_left = 0
        if repeat_left == action_repeat:
            t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        load_model=None,
        action_bias=0.4,
        action_repeat=0,
        action_repeat_rand=False,
        warmup_frames=0,
        target_kl=0.01,
        vf_loss_mult=1,
        vfloss_optim_stepsize=0.003,
        vfloss_optim_batchsize=8,
        vfloss_optim_epochs=10
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    # Not sure why they anneal clip and learning rate with the same parameter
    #clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen
    losses = [pol_surr, pol_entpen, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    lossandgrad_vfloss = U.function([ob, ac, atarg, ret], [vf_loss] + [U.flatgrad(vf_loss, var_list)])
    adam_vfloss = MpiAdam(var_list, epsilon=adam_epsilon)
    compute_vfloss = U.function([ob, ac, atarg, ret], [vf_loss])

    U.initialize()
    adam.sync()
    adam_vfloss.sync()

    if load_model:
        logger.log('Loading model: %s' % load_model)
        pi.load(load_model)


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, action_bias=action_bias, action_repeat=action_repeat, action_repeat_rand=action_repeat_rand, warmup_frames=warmup_frames)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    ep_rew_file = None
    if MPI.COMM_WORLD.Get_rank()==0:
        import wandb
        ep_rew_file = open(os.path.join(wandb.run.dir, 'episode_rewards.jsonl'), 'w')
        checkpoint_dir = 'checkpoints-%s' % wandb.run.id
        os.mkdir(checkpoint_dir)

    cur_lrmult = 1.0
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        elif schedule == 'target_kl':
            pass
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.next()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                result = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                newlosses = result[:-1]
                g = result[-1]
                adam.update(g, optim_stepsize * cur_lrmult) 
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        # vfloss optimize
        logger.log("Optimizing value function...")
        logger.log(fmt_row(13, ['vf']))
        for _ in range(vfloss_optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(vfloss_optim_batchsize):
                result = lossandgrad_vfloss(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"])
                newlosses = result[:-1]
                g = result[-1]
                adam_vfloss.update(g, vfloss_optim_stepsize) 
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            newlosses += compute_vfloss(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"])
            losses.append(newlosses)            
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names + ['vf']):
            logger.record_tabular("loss_"+name, lossval)
        # check kl
        if schedule == 'target_kl':
            if meanlosses[2] > target_kl * 1.1:
                cur_lrmult /= 1.5
            elif meanlosses[2] < target_kl / 1.1:
                cur_lrmult *= 1.5
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        if rewbuffer:
            logger.record_tabular('CurLrMult', cur_lrmult)
            logger.record_tabular('StepSize', optim_stepsize * cur_lrmult)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMax", np.max(rewbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpRewMin", np.min(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            time_elapsed = time.time() - tstart
            logger.record_tabular("TimeElapsed", time_elapsed)
            if MPI.COMM_WORLD.Get_rank()==0:
                import wandb
                ep_rew_file.write('%s\n' % json.dumps({
                    'TimeElapsed': time_elapsed,
                    'Rewards': rews}))
                ep_rew_file.flush()
                data = logger.Logger.CURRENT.name2val
                wandb.run.history.add(data)
                summary_data = {}
                for k, v in data.iteritems():
                    if 'Rew' in k:
                        summary_data[k] = v
                wandb.run.summary.update(summary_data)
                pi.save(os.path.join(checkpoint_dir, 'model-%s.ckpt' % (iters_so_far - 1)))

                logger.dump_tabular()
        else:
            logger.log('No episodes complete yet')

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
