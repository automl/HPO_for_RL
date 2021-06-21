import os
import shutil


class GarbageCollector:
    def __init__(self, data_path):
        self.data_path = data_path

    def collect(self, member_id, min_timestep, elites):
        if min_timestep - 1 < 0:
            return
        else:
            for i in range(min_timestep-1):
                if member_id in [elite.member_id for elite in elites] and i in [elite.time_step for elite in elites]:
                    continue
                member_path = os.path.join(self.data_path, str(member_id), str(i))
                for file_name in ["state_dict.npz", "traj_acs.json", "traj_obs.json", "traj_rews.json"]:
                    file_path = os.path.join(member_path, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    #else:
                    #    print("Can not delete the file in path %s as it doesn't exists" %file_path)

    def _get_integer_dirs(self, path):
        return [int(i) for i in os.listdir(path) if i.isdigit()]
