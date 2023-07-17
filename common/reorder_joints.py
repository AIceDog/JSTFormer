def Reorder_Joints(split, dataset, poses, poses_2d):
    if split == 'train':
        if dataset == 'h36m':
            # spine : 0, 7, 8, 9, 10
            # lArm : 14, 15, 16
            # rLeg : 4, 5, 6
            # rArm : 11, 12, 13
            # lLeg : 1, 2, 3
            joint_idx = [0, 7, 8, 9, 10, 14, 15, 16, 4, 5, 6, 11, 12, 13, 1, 2, 3]

            for key in poses:
                poses[key] = poses[key][:, joint_idx, :]
                
            for key in poses_2d:
                poses_2d[key] = poses_2d[key][:, joint_idx, :]
                
    elif split == 'test':
        if dataset == 'h36m':
            # spine : 0, 7, 8, 9, 10
            # lArm : 14, 15, 16
            # rLeg : 4, 5, 6
            # rArm : 11, 12, 13
            # lLeg : 1, 2, 3
            joint_idx = [0, 7, 8, 9, 10, 14, 15, 16, 4, 5, 6, 11, 12, 13, 1, 2, 3]
            
            for key in poses:
                poses[key] = poses[key][:, joint_idx, :]
                
            for key in poses_2d:
                poses_2d[key] = poses_2d[key][:, joint_idx, :]
                
    return poses, poses_2d