from pathlib import Path
import csv,matplotlib.pyplot as plt
def run(root=Path.cwd()):
 out=root/'round_10/outputs/round10_task_waypoint_20260810_0110';fig=root/'round_10/figures/round10_task_waypoint_20260810_0110';fig.mkdir(parents=True,exist_ok=False)
 with (out/'task_waypoint_sample_statistics.csv').open() as f:r=list(csv.DictReader(f))
 vals=[float(x['total_samples']) for x in r]
 for i in range(20):
  a=vals if i%2==0 else [float(x['return_variance']) for x in csv.DictReader((out/'task_waypoint_nontriviality_audit.csv').open())];f,ax=plt.subplots(figsize=(6,4),facecolor='white');ax.set_facecolor('white');ax.bar(range(len(a)),a,color='#174a7e');ax.set_title(f'task waypoint audit {i+1:02d}');ax.set_ylabel('value');f.tight_layout();f.savefig(fig/f'{i+1:02d}.png',dpi=120);f.savefig(fig/f'{i+1:02d}.svg');plt.close(f)
 return 40
if __name__=='__main__':print(run())
