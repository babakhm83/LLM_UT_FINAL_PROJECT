import matplotlib.pyplot as plt
import numpy as np
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

def plot_final_training_loss():
    """Plot the final training loss data"""
    
    # Training data for Individual News Dataset Final Training
    training_data = """2	3.142800
4	3.408000
6	3.178600
8	4.340900
10	3.610200
12	3.621900
14	3.280300
16	2.812000
18	2.589300
20	2.300900
22	2.213000
24	1.940700
26	1.853900
28	1.843700
30	1.749300
32	1.679800
34	1.662800
36	1.587100
38	1.540400
40	1.561900
42	1.519900
44	1.517900
46	1.515800
48	1.448300
50	1.464300
52	1.412900
54	1.404900
56	1.391200
58	1.334000
60	1.316400
62	1.300400
64	1.263400
66	1.226000
68	1.271400
70	1.216100
72	1.205100
74	1.185200
76	1.166900
78	1.161900
80	1.189000
82	1.130500
84	1.160300
86	1.129100
88	1.109900
90	1.106500
92	1.099300
94	1.124900
96	1.064500
98	1.070200
100	1.068100
102	1.044600
104	1.052200
106	1.048500
108	1.074800
110	1.043100
112	1.049400
114	1.033500
116	1.045100
118	1.005700
120	1.009900
122	0.998100
124	0.994300
126	0.981200
128	1.028800
130	0.994700
132	1.035100
134	0.995000
136	0.978000
138	0.964600
140	0.997600
142	0.955800
144	0.980400
146	0.998300
148	0.961400
150	0.955600
152	0.956200
154	0.940000
156	0.946400
158	0.917300
160	0.935500
162	0.934300
164	0.925000
166	0.928600
168	0.963000
170	0.923200
172	0.935100
174	0.967400
176	0.930500
178	0.926700
180	0.905300
182	0.895700
184	0.930500
186	0.910500
188	0.923200
190	0.914800
192	0.967900
194	0.904400
196	0.926000
198	0.942700
200	0.899000
202	0.886300
204	0.896100
206	0.911100
208	0.894300
210	0.890400
212	0.918500
214	0.900600
216	0.876800
218	0.896400
220	0.906100
222	0.893000
224	0.893000
226	0.895800
228	0.904000
230	0.889700
232	0.900000
234	0.879000
236	0.874600
238	0.845200
240	0.881800
242	0.906000
244	0.882800
246	0.871500
248	0.853800
250	0.885300
252	0.884600
254	0.881000
256	0.859700
258	0.873400
260	0.872200
262	0.869400
264	0.884100
266	0.888400
268	0.849200
270	0.876700
272	0.851600
274	0.870300
276	0.851800
278	0.839300
280	0.851600
282	0.853800
284	0.852600
286	0.878400
288	0.865600
290	0.864000
292	0.854600
294	0.844800
296	0.827600
298	0.839800
300	0.836400
302	0.881700
304	0.827400
306	0.820600
308	0.862500
310	0.806000
312	0.806700
314	0.858800
316	0.824300
318	0.827100
320	0.855300
322	0.823000
324	0.818100
326	0.818300
328	0.825200
330	0.807500
332	0.841200
334	0.839300
336	0.810700
338	0.860000
340	0.773500
342	0.805400
344	0.800500
346	0.811600
348	0.800100
350	0.833800
352	0.820300
354	0.791300
356	0.808100
358	0.777000
360	0.812200
362	0.823100
364	0.802900
366	0.777200
368	0.815200
370	0.804600
372	0.790100
374	0.813000
376	0.817800
378	0.809500
380	0.798300
382	0.809000
384	0.799300
386	0.790500
388	0.828300
390	0.804300
392	0.804800
394	0.791300
396	0.796300
398	0.783800
400	0.791100
402	0.786700
404	0.781600
406	0.813900
408	0.816700
410	0.789900
412	0.784500
414	0.809500
416	0.810000
418	0.785800
420	0.792100
422	0.808200
424	0.781600
426	0.805500
428	0.813100
430	0.788300
432	0.767400
434	0.791400
436	0.820900
438	0.817600
440	0.832800
442	0.805400
444	0.792000
446	0.806400
448	0.791200
450	0.785400
452	0.790000
454	0.780400
456	0.820400
458	0.818100
460	0.800600
462	0.749600
464	0.774500
466	0.794400
468	0.785200
470	0.792700
472	0.790200
474	0.801500
476	0.786400
478	0.766800
480	0.802300
482	0.773500
484	0.787100
486	0.822000
488	0.762700
490	0.818600
492	0.806000
494	0.772000
496	0.771600
498	0.813600
500	0.762500
502	0.772300
504	0.792800
506	0.790700"""

    # Parse the training data
    steps, losses = parse_training_data(training_data)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with multiple subplots for comprehensive analysis
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Raw training loss
    plt.subplot(2, 3, 1)
    plt.plot(steps, losses, color='#2E86AB', linewidth=2, alpha=0.8, marker='o', markersize=1)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('SFT Individual News Dataset Final - Raw Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Smoothed training loss
    plt.subplot(2, 3, 2)
    smoothed_losses = smooth_losses(losses, window_size=20)
    plt.plot(steps, smoothed_losses, color='#A23B72', linewidth=3, alpha=0.9)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss (Smoothed)', fontsize=12)
    plt.title('SFT Individual News Dataset Final - Smoothed Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Loss improvement over time
    plt.subplot(2, 3, 3)
    if len(losses) > 10:
        initial_avg = np.mean(losses[:10])
        final_avg = np.mean(losses[-10:])
        improvement_pct = ((initial_avg - final_avg) / initial_avg) * 100
        
        plt.plot(steps, [(losses[0] - loss) / losses[0] * 100 for loss in losses], 
                color='#F18F01', linewidth=2.5, alpha=0.9)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss Improvement (%)', fontsize=12)
        plt.title(f'Training Improvement - {improvement_pct:.1f}% Total Reduction', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss distribution histogram
    plt.subplot(2, 3, 4)
    plt.hist(losses, bins=40, alpha=0.7, color='#C73E1D', edgecolor='black')
    plt.xlabel('Training Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Loss Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Learning phases analysis
    plt.subplot(2, 3, 5)
    # Divide into phases
    phase_size = len(steps) // 4
    phases = ['Early (0-25%)', 'Mid-Early (25-50%)', 'Mid-Late (50-75%)', 'Final (75-100%)']
    phase_avgs = []
    
    for i in range(4):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < 3 else len(losses)
        phase_avg = np.mean(losses[start_idx:end_idx])
        phase_avgs.append(phase_avg)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = plt.bar(phases, phase_avgs, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Loss by Training Phase', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, avg in zip(bars, phase_avgs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Convergence analysis (last 200 steps)
    plt.subplot(2, 3, 6)
    if len(steps) > 200:
        conv_steps = steps[-200:]
        conv_losses = losses[-200:]
        
        # Calculate trend line
        z = np.polyfit(range(len(conv_losses)), conv_losses, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(conv_losses)))
        
        plt.plot(conv_steps, conv_losses, color='#2E86AB', linewidth=2, alpha=0.8, label='Actual Loss')
        plt.plot(conv_steps, trend_line, color='#C73E1D', linewidth=3, alpha=0.9, 
                linestyle='--', label=f'Trend (slope: {z[0]:.6f})')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Convergence Analysis (Final 200 Steps)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("./")
    filename = "SFT_bitcoin-individual-news-dataset_t_train__effect_longer_final.png"
    plt.savefig(output_dir / filename, 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Create a simple version too
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simple raw plot
    ax1.plot(steps, losses, color='#2E86AB', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Training Loss', fontsize=14)
    ax1.set_title('SFT Individual News Dataset Final Training - Raw Loss', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.4)
    ax1.set_yscale('log')
    
    # Simple smoothed plot  
    ax2.plot(steps, smoothed_losses, color='#A23B72', linewidth=3, alpha=0.9)
    ax2.set_xlabel('Training Steps', fontsize=14)
    ax2.set_ylabel('Training Loss (Smoothed)', fontsize=14)
    ax2.set_title('SFT Individual News Dataset Final Training - Smoothed Loss', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.4)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save simple version
    simple_filename = "SFT_bitcoin-individual-news-dataset_t_train__effect_longer_final_simple.png"
    plt.savefig(output_dir / simple_filename, 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Training loss plots saved successfully!")
    print("📊 Generated files:")
    print(f"   - {filename}")
    print(f"   - {simple_filename}")
    print(f"\n📈 Training Summary:")
    print(f"   - Total steps: {len(steps)}")
    print(f"   - Initial loss: {losses[0]:.4f}")
    print(f"   - Final loss: {losses[-1]:.4f}")
    print(f"   - Total improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print(f"   - Minimum loss achieved: {min(losses):.4f}")

if __name__ == "__main__":
    plot_final_training_loss()
