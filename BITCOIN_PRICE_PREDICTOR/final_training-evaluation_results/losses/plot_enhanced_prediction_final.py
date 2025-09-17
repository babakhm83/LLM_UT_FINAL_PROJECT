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

def plot_enhanced_prediction_final():
    """Plot the enhanced prediction dataset final training loss data"""
    
    # Training data for Enhanced Prediction Dataset Final Training
    training_data = """2	21.379600
4	2.519900
6	20.921600
8	16.422500
10	2.524000
12	2.485600
14	2.435200
16	8.159900
18	2.359800
20	2.317800
22	2.304600
24	2.909300
26	2.777400
28	2.732500
30	2.622500
32	2.526900
34	2.049400
36	1.992300
38	2.239600
40	2.191100
42	1.834500
44	2.055800
46	2.482400
48	1.704400
50	1.741400
52	1.884900
54	1.657800
56	1.658100
58	1.627600
60	1.962800
62	1.900700
64	1.525200
66	1.679300
68	1.770900
70	1.632700
72	1.610800
74	1.607600
76	1.604200
78	1.571400
80	1.542800
82	1.538200
84	1.533300
86	1.510100
88	1.597500
90	1.577600
92	1.558600
94	1.464000
96	1.472800
98	1.463300
100	1.343300
102	1.377500
104	1.675700
106	1.322200
108	1.340100
110	1.381600
112	1.317900
114	1.376000
116	1.317700
118	1.277000
120	1.407500
122	1.293200
124	1.280900
126	1.292800
128	1.282000
130	1.319200
132	1.288100
134	1.260800
136	1.265700
138	1.274300
140	1.306800
142	1.347400
144	1.304000
146	1.253800
148	1.268200
150	1.234800
152	1.246900
154	1.256500
156	1.281700
158	1.230400
160	1.227900
162	1.269000
164	1.362100
166	1.279000
168	1.368900
170	1.261300
172	1.271900
174	1.227600
176	1.280700
178	1.219200
180	1.248600
182	1.239400
184	1.241800
186	1.302100
188	1.236900
190	1.233800
192	1.241100
194	1.213600
196	1.242700
198	1.214300
200	1.218800
202	1.198800
204	1.201100
206	1.215300
208	1.216400
210	1.199300
212	1.239700
214	1.197800
216	1.221400
218	1.221800
220	1.195600
222	1.210900
224	1.227100
226	1.208100
228	1.215700
230	1.199400
232	1.197600
234	1.227400
236	1.213200
238	1.212600
240	1.221700
242	1.238600
244	1.222900
246	1.202700
248	1.199100
250	1.228500
252	1.206700
254	1.200600
256	1.199600
258	1.190400
260	1.215500
262	1.217400
264	1.211700
266	1.202800
268	1.212200
270	1.216300
272	1.217400
274	1.198800
276	1.190600
278	1.218700
280	1.190900
282	1.226500
284	1.174200
286	1.202900
288	1.193800
290	1.211700
292	1.175800
294	1.204700
296	1.199600
298	1.194700
300	1.201900
302	1.194700
304	1.196300
306	1.200300
308	1.217300
310	1.176800
312	1.191700
314	1.202000
316	1.182900
318	1.190600
320	1.214800
322	1.190200
324	1.186400
326	1.198600
328	1.194200
330	1.186300
332	1.185000
334	1.227800
336	1.187300
338	1.200700
340	1.218700
342	1.209400
344	1.173500
346	1.190500
348	1.208000
350	1.200400
352	1.203100
354	1.199900
356	1.198700
358	1.150300
360	1.198000
362	1.167100
364	1.191800
366	1.205400
368	1.210300
370	1.193900
372	1.179900
374	1.192700
376	1.204800
378	1.220800
380	1.199300
382	1.190900
384	1.186800
386	1.197000
388	1.195700
390	1.209700
392	1.177700
394	1.189000
396	1.204600
398	1.221700
400	1.215100
402	1.182800
404	1.193200
406	1.202700
408	1.177100
410	1.148600
412	1.188900
414	1.217600
416	1.201800
418	1.198500
420	1.172700
422	1.205500
424	1.183500
426	1.221500
428	1.221500
430	1.202900
432	1.202600
434	1.195500
436	1.172000
438	1.221000
440	1.215500
442	1.206300
444	1.212800
446	1.190300
448	1.176900
450	1.199900
452	1.162500
454	1.177100
456	1.218500
458	1.185300
460	1.216200"""

    # Parse the training data
    steps, losses = parse_training_data(training_data)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with multiple subplots for comprehensive analysis
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Raw training loss
    plt.subplot(2, 3, 1)
    plt.plot(steps, losses, color='#E85A4F', linewidth=2, alpha=0.8, marker='o', markersize=1)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Enhanced Prediction Dataset Final - Raw Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Smoothed training loss
    plt.subplot(2, 3, 2)
    smoothed_losses = smooth_losses(losses, window_size=20)
    plt.plot(steps, smoothed_losses, color='#C73E1D', linewidth=3, alpha=0.9)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Training Loss (Smoothed)', fontsize=12)
    plt.title('Enhanced Prediction Dataset Final - Smoothed Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Loss improvement over time
    plt.subplot(2, 3, 3)
    if len(losses) > 10:
        # Filter out the extreme outliers for better visualization
        filtered_losses = [l for l in losses if l < 10]  # Remove extreme spikes
        initial_avg = np.mean(filtered_losses[:10]) if len(filtered_losses) >= 10 else np.mean(losses[:10])
        final_avg = np.mean(filtered_losses[-10:]) if len(filtered_losses) >= 10 else np.mean(losses[-10:])
        improvement_pct = ((initial_avg - final_avg) / initial_avg) * 100
        
        # Use filtered losses for improvement calculation
        base_loss = filtered_losses[0] if filtered_losses else losses[0]
        improvement_values = []
        for i, loss in enumerate(losses):
            if loss < 10:  # Only calculate improvement for reasonable losses
                improvement = ((base_loss - loss) / base_loss) * 100
                improvement_values.append(improvement)
            else:
                # For outliers, use the previous value or 0
                improvement_values.append(improvement_values[-1] if improvement_values else 0)
        
        plt.plot(steps, improvement_values, 
                color='#D2691E', linewidth=2.5, alpha=0.9)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss Improvement (%)', fontsize=12)
        plt.title(f'Training Improvement - {improvement_pct:.1f}% Total Reduction', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss distribution histogram (filtered)
    plt.subplot(2, 3, 4)
    # Filter losses for better histogram visualization
    histogram_losses = [l for l in losses if l < 5]  # Remove extreme outliers
    plt.hist(histogram_losses, bins=40, alpha=0.7, color='#CD853F', edgecolor='black')
    plt.xlabel('Training Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Loss Distribution (Filtered < 5)', fontsize=14, fontweight='bold')
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
        # Filter out extreme outliers for phase averages
        phase_losses = [losses[j] for j in range(start_idx, end_idx) if losses[j] < 10]
        phase_avg = np.mean(phase_losses) if phase_losses else np.mean(losses[start_idx:end_idx])
        phase_avgs.append(phase_avg)
    
    colors = ['#E85A4F', '#C73E1D', '#D2691E', '#CD853F']
    bars = plt.bar(phases, phase_avgs, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Loss by Training Phase', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, avg in zip(bars, phase_avgs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Convergence analysis (last 100 steps)
    plt.subplot(2, 3, 6)
    if len(steps) > 100:
        conv_steps = steps[-100:]
        conv_losses = losses[-100:]
        
        # Calculate trend line
        z = np.polyfit(range(len(conv_losses)), conv_losses, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(conv_losses)))
        
        plt.plot(conv_steps, conv_losses, color='#E85A4F', linewidth=2, alpha=0.8, label='Actual Loss')
        plt.plot(conv_steps, trend_line, color='#8B0000', linewidth=3, alpha=0.9, 
                linestyle='--', label=f'Trend (slope: {z[0]:.6f})')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Convergence Analysis (Final 100 Steps)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_dir = Path("./")
    filename = "SFT_bitcoin-enhanced-prediction-dataset-with-local-comprehensive-news_train_out_final.png"
    plt.savefig(output_dir / filename, 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Create a simple version too
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simple raw plot
    ax1.plot(steps, losses, color='#E85A4F', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Training Loss', fontsize=14)
    ax1.set_title('Enhanced Prediction Dataset Final - Raw Loss', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.4)
    ax1.set_yscale('log')
    
    # Simple smoothed plot  
    ax2.plot(steps, smoothed_losses, color='#C73E1D', linewidth=3, alpha=0.9)
    ax2.set_xlabel('Training Steps', fontsize=14)
    ax2.set_ylabel('Training Loss (Smoothed)', fontsize=14)
    ax2.set_title('Enhanced Prediction Dataset Final - Smoothed Loss', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.4)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save simple version
    simple_filename = "SFT_bitcoin-enhanced-prediction-dataset-with-local-comprehensive-news_train_out_final_simple.png"
    plt.savefig(output_dir / simple_filename, 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Calculate statistics
    filtered_losses = [l for l in losses if l < 10]  # Remove extreme outliers for stats
    min_loss = min(filtered_losses) if filtered_losses else min(losses)
    max_loss = max(filtered_losses) if filtered_losses else max([l for l in losses if l < 10])
    
    print("✅ Enhanced Prediction Dataset training loss plots saved successfully!")
    print("📊 Generated files:")
    print(f"   - {filename}")
    print(f"   - {simple_filename}")
    print(f"\n📈 Training Summary:")
    print(f"   - Total steps: {len(steps)} (currently at step 460, epoch 1.60/2)")
    print(f"   - Initial loss: {losses[0]:.4f}")
    print(f"   - Current final loss: {losses[-1]:.4f}")
    print(f"   - Minimum loss achieved: {min_loss:.4f}")
    print(f"   - Maximum reasonable loss: {max_loss:.4f}")
    
    # Calculate improvement excluding outliers
    if filtered_losses and len(filtered_losses) > 10:
        initial_filtered = np.mean([l for l in losses[:10] if l < 10])
        final_filtered = np.mean([l for l in losses[-10:] if l < 10])
        improvement = ((initial_filtered - final_filtered) / initial_filtered) * 100
        print(f"   - Training improvement (filtered): {improvement:.1f}%")
    
    print(f"   - Note: Dataset shows some loss spikes, likely due to complex news data")

if __name__ == "__main__":
    plot_enhanced_prediction_final()
