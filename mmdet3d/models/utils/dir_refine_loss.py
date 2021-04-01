import torch
import numpy as np

def huber_loss(error, delta=1.0):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss

def get_dir_refine_loss(heading_angle, gt_heading, num_heading_bin):
    gt_heading = gt_heading % (2*np.pi)
    heading_angle = heading_angle % (2*np.pi)

    batch_size, num_proposal = heading_angle.shape

    heading_delta = gt_heading - heading_angle
    heading_delta_neg = (2*np.pi) + heading_delta
    heading_delta_pos = heading_delta - (2*np.pi)

    heading_delta_neg_indicator = torch.zeros((batch_size, num_proposal)).cuda()
    heading_delta_neg_indicator[heading_angle < -np.pi] = 1

    heading_delta_pos_indicator = torch.zeros((batch_size, num_proposal)).cuda()
    heading_delta_pos_indicator[heading_delta > np.pi] = 1

    heading_delta_dont_care_indicator = torch.zeros((batch_size, num_proposal)).cuda()
    heading_delta_dont_care_indicator[(heading_delta >= -np.pi) * (heading_delta <= np.pi)] = 1

    heading_delta = heading_delta*heading_delta_dont_care_indicator + \
                    heading_delta_neg*heading_delta_neg_indicator + \
                    heading_delta_pos*heading_delta_pos_indicator
    heading_loss = huber_loss(heading_delta, delta=np.pi/num_heading_bin)  # (B, N)
    return heading_loss