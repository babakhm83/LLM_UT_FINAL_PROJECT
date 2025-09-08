import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

def parse_training_data(data_string):
    """Parse training log data and extract steps and losses"""
    lines = data_string.strip().split('\n')
    steps = []
    losses = []
    
    for line in lines:
        if '\t' in line and line.count('\t') == 1:
            try:
                step_str, loss_str = line.split('\t')
                step = int(step_str.strip())
                loss = float(loss_str.strip())
                
                # Filter out zero losses (they indicate padding/errors)
                if loss > 0:
                    steps.append(step)
                    losses.append(loss)
            except (ValueError, IndexError):
                continue
    
    return steps, losses

def smooth_losses(losses, window_size=10):
    """Apply moving average smoothing to losses"""
    if len(losses) < window_size:
        return losses
    
    smoothed = []
    for i in range(len(losses)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(losses), i + window_size // 2 + 1)
        smoothed.append(np.mean(losses[start_idx:end_idx]))
    
    return smoothed

def plot_training_losses():
    """Plot all training losses with proper formatting"""
    
    # Training data for each experiment - ENHANCED PREDICTION DATASET
    enhanced_data = """2	21.379600
4	2.514400
6	13.802600
8	4.983500
10	2.397000
12	2.316900
14	2.212100
16	2.591100
18	1.977400
20	1.839100
22	1.728900
24	1.867000
26	1.727600
28	1.650600
30	1.573200
32	1.506200
34	1.393000
36	1.365400
38	1.366200
40	1.344000
42	1.280900
44	1.291100
46	1.309500
48	1.214900
50	1.252200
52	1.231100
54	1.213800
56	1.214900
58	1.212000
60	1.223900
62	1.207500
64	1.202600
66	1.212600
68	1.166000
70	1.215000
72	1.186800
74	1.197100
76	1.168400
78	1.198100
80	1.183800
82	1.203200
84	1.193700
86	1.196500
88	1.190800
90	1.198300
92	1.184200
94	1.187100
96	1.207200
98	1.208800
100	1.184700
102	1.204100
104	1.219400
106	1.178800
108	1.197900
110	1.183600
112	1.188300
114	1.183000
116	1.195700
118	1.160500
120	1.178700
122	1.184900
124	1.177900
126	1.187800
128	1.196300
130	1.213500
132	1.200000
134	1.174700
136	1.190700
138	1.203400
140	1.176400
142	1.169000
144	1.178700
146	1.190800
148	1.200100
150	1.177100
152	1.180900
154	1.206600
156	1.187900
158	1.175800
160	1.174100
162	1.185500
164	1.193700
166	1.194600
168	1.189800
170	1.182200
172	1.175100
174	1.182600
176	1.213000
178	1.183300
180	1.197400
182	1.193100
184	1.190100
186	1.204500
188	1.189900
190	1.202900
192	1.206300
194	1.186100
196	1.214300
198	1.193900
200	1.190900
202	1.166900
204	1.172400
206	1.175500
208	1.197200
210	1.168900
212	1.190400
214	1.179000
216	1.174800
218	1.197500
220	1.175900
222	1.188600
224	1.198000
226	1.185100
228	1.194200
230	1.171300
232	1.176800
234	1.187800
236	1.194500
238	1.175900
240	1.207000
242	1.208700
244	1.193700
246	1.186400
248	1.172600
250	1.212700
252	1.188400
254	1.169800
256	1.168100
258	1.170800
260	1.199500
262	1.189800
264	1.189800
266	1.187000
268	1.190800
270	1.206000
272	1.215400
274	1.182500
276	1.183100
278	1.202100
280	1.179900
282	1.202800
284	1.167700
286	1.209900
288	1.195000
290	1.200900
292	1.173600
294	1.203100
296	1.198900
298	1.186200
300	1.196200
302	1.182200
304	1.214000
306	1.207500
308	1.229900
310	1.208000
312	1.228600
314	1.245400
316	1.206100
318	1.215500
320	1.236000
322	1.253000
324	1.244800
326	1.247800
328	1.256900
330	1.270700
332	1.289300
334	1.371600
336	1.367700
338	1.387100
340	1.426600
342	1.503500
344	1.532000
346	1.585400
348	1.616100
350	1.650500
352	1.654600
354	1.648300
356	1.682100
358	1.703600
360	1.713300
362	1.724600
364	1.765300
366	1.816300
368	1.864500
370	1.904000
372	1.925300
374	1.989700
376	2.027800
378	2.039300
380	2.061600
382	2.073500
384	2.178900
386	2.145600
388	2.172800
390	2.232600
392	2.296300
394	2.409700
396	2.440400
398	2.560300
400	2.695000
402	2.914900
404	3.427600
406	3.839300
408	4.228000
410	4.564100
412	4.851000
414	5.047600
416	5.131600
418	5.495500
420	5.518100
422	5.675000
424	5.616100
426	5.619100
428	5.661600
430	5.697100
432	5.787500
434	5.730500
436	5.789100
438	6.014900
440	5.983500
442	6.145000
444	6.192100
446	6.242800
448	6.236600
450	6.518400
452	6.358200
454	6.416500
456	6.568100
458	6.466300"""

    # INDIVIDUAL NEWS DATASET 
    individual_news_data = """2	4.159100
4	3.935700
6	2.052500
8	0.940800
10	0.692600
12	0.520700
14	0.479400
16	0.430100
18	0.405500
20	0.407600
22	0.388000
24	0.396000
26	0.381300
28	0.380100
30	0.364400
32	0.357600
34	0.351000
36	0.345900
38	0.358900
40	0.338000
42	0.342900
44	0.328700
46	0.328100
48	0.336500
50	0.338600
52	0.321200
54	0.307800
56	0.316300
58	0.304800
60	0.296400
62	0.297300
64	0.291200
66	0.292600
68	0.300200
70	0.302200
72	0.293700
74	0.299800
76	0.298900
78	0.298400
80	0.290700
82	0.292200
84	0.294600
86	0.287800
88	0.293600
90	0.285500
92	0.289100
94	0.293900
96	0.282400
98	0.289400
100	0.279400
102	0.261200
104	0.257700
106	0.260700
108	0.256900
110	0.238400
112	0.258600
114	0.250500
116	0.246700
118	0.250500
120	0.250100
122	0.252600
124	0.249100
126	0.239800
128	0.248900
130	0.255100
132	0.254800
134	0.243800
136	0.247400
138	0.249000
140	0.246300
142	0.246900
144	0.246200
146	0.251900
148	0.245600
150	0.244400
152	0.228600
154	0.224200
156	0.210900
158	0.220900
160	0.217100
162	0.223000
164	0.221000
166	0.211900
168	0.211800
170	0.208700
172	0.217400
174	0.208900
176	0.203200
178	0.217000
180	0.212600
182	0.212900
184	0.223200
186	0.207400
188	0.209100
190	0.209700
192	0.220700
194	0.213600
196	0.213200
198	0.209000
200	0.208900"""

    # INDIVIDUAL NEWS EXTENDED DATASET (truncated to first 1500 steps for visibility)
    individual_extended_data = """2	18.723800
4	19.950200
6	21.542700
8	20.241400
10	17.904300
12	13.262700
14	9.674600
16	9.570500
18	7.179000
20	5.732600
22	3.281800
24	2.673500
26	2.233600
28	1.933700
30	1.703100
32	1.548600
34	1.403300
36	1.344000
38	1.252500
40	1.165900
42	1.062700
44	0.998900
46	0.987500
48	0.949300
50	0.938200
52	0.977600
54	0.948700
56	0.898100
58	0.931300
60	0.907300
62	0.859800
64	0.861400
66	0.872300
68	0.832400
70	0.834600
72	0.827000
74	0.804000
76	0.789200
78	0.781600
80	0.803100
82	0.785900
84	0.777600
86	0.832800
88	0.812100
90	0.767600
92	0.780000
94	0.751300
96	0.761200
98	0.748200
100	0.754400
200	0.661000
300	0.614400
400	0.564500
500	0.573500
600	0.536600
700	0.580100
800	0.583500
900	0.558400
1000	0.532300
1100	0.565600
1200	0.570600
1300	0.572800
1400	0.545900
1500	0.608500"""

    # Create training data dictionary
    training_data = {
        "Enhanced Prediction Dataset (4 epochs)": enhanced_data,
        "Individual News Dataset (4 epochs)": individual_news_data,
        "Individual News Extended (4 epochs)": individual_extended_data
    }
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: All training curves together
    plt.subplot(2, 2, 1)
    for i, (name, data) in enumerate(training_data.items()):
        steps, losses = parse_training_data(data)
        if steps and losses:
            plt.plot(steps, losses, label=name, color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Bitcoin Price Prediction - All Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Smoothed version
    plt.subplot(2, 2, 2)
    for i, (name, data) in enumerate(training_data.items()):
        steps, losses = parse_training_data(data)
        if steps and losses:
            smoothed_losses = smooth_losses(losses, window_size=20)
            plt.plot(steps, smoothed_losses, label=f"{name} (smoothed)", 
                    color=colors[i % len(colors)], linewidth=3, alpha=0.9)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss (Smoothed)', fontsize=12)
    plt.title('Bitcoin Price Prediction - Smoothed Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Focus on convergence (later steps only)
    plt.subplot(2, 2, 3)
    for i, (name, data) in enumerate(training_data.items()):
        steps, losses = parse_training_data(data)
        if steps and losses:
            # Only show steps after 100 to focus on convergence
            convergence_steps = [s for s in steps if s >= 100]
            convergence_losses = [losses[j] for j, s in enumerate(steps) if s >= 100]
            
            if convergence_steps:
                plt.plot(convergence_steps, convergence_losses, 
                        label=name, color=colors[i % len(colors)], 
                        linewidth=2, alpha=0.8, marker='o', markersize=2)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Bitcoin Price Prediction - Convergence Behavior (Steps ≥ 100)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss distribution histogram
    plt.subplot(2, 2, 4)
    all_losses = []
    labels = []
    for name, data in training_data.items():
        steps, losses = parse_training_data(data)
        if losses:
            # Filter out very high losses for better visualization
            filtered_losses = [l for l in losses if l < 10]
            if filtered_losses:
                all_losses.append(filtered_losses)
                labels.append(name)
    
    if all_losses:
        plt.hist(all_losses, label=labels, alpha=0.7, bins=30, color=colors[:len(all_losses)])
    
    plt.xlabel('Training Loss', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Bitcoin Price Prediction - Loss Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_dir = Path("./")
    plt.savefig(output_dir / "bitcoin_training_losses_comprehensive_comparison.png", 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # Close to free memory
    
    # Create individual plots for each experiment
    for i, (name, data) in enumerate(training_data.items()):
        steps, losses = parse_training_data(data)
        if steps and losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Raw loss plot
            ax1.plot(steps, losses, color=colors[i % len(colors)], linewidth=2, alpha=0.8)
            ax1.set_xlabel('Training Steps', fontsize=12)
            ax1.set_ylabel('Training Loss', fontsize=12)
            ax1.set_title(f'{name} - Raw Training Loss', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Smoothed loss plot
            smoothed_losses = smooth_losses(losses, window_size=15)
            ax2.plot(steps, smoothed_losses, color=colors[i % len(colors)], linewidth=3, alpha=0.9)
            ax2.set_xlabel('Training Steps', fontsize=12)
            ax2.set_ylabel('Training Loss (Smoothed)', fontsize=12)
            ax2.set_title(f'{name} - Smoothed Training Loss', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            plt.tight_layout()
            
            # Clean filename
            clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            plt.savefig(output_dir / f"bitcoin_training_loss_{clean_name}.png", 
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
    
    print("✅ All plots saved successfully!")
    print("📊 Generated files:")
    print("   - bitcoin_training_losses_comprehensive_comparison.png")
    for name, _ in training_data.items():
        clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        print(f"   - bitcoin_training_loss_{clean_name}.png")

if __name__ == "__main__":
    plot_training_losses()
