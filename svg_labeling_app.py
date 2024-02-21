import gradio as gr
import pandas as pd
from torchvision.transforms import ToPILImage
from thesis.utils import svg2paths2, raster, disvg


# Load your CSV file
df_path = "/scratch2/moritz_data/example_dafont.csv"  # Update this path
df = pd.read_csv(df_path)

# Ensure the manual_label column exists
if 'manual_label' not in df.columns:
    df['manual_label'] = None

# Function to load and display the SVG file as a base64-encoded image
def show_svg(svg_path):
    # with open(svg_path, 'r') as file:
    #     svg_content = file.read()
    # svg_base64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
    # return f"data:image/svg+xml;base64,{svg_base64}"
    return svg_file_path_to_image(svg_path)

# Function to handle the labeling
def label_svg(index, label):
    df.at[index, 'manual_label'] = label
    # Save the DataFrame periodically
    if df.index[df['manual_label'].notnull()].shape[0] % 10 == 0:
        df.to_csv(df_path, index=False, escapechar='\\')
    return "Label saved! Continue labeling."

# Function to manually save the DataFrame
def save_df():
    df.to_csv(df_path, index=False, escapechar='\\')
    return "DataFrame saved manually."

def svg_file_path_to_image(path):
    paths, attributes, svg_attributes = svg2paths2(path)
    return_tensor = raster(disvg(paths, stroke_widths=[0.5]*len(paths),paths2Drawing=True), out_h=224, out_w = 224)
    return ToPILImage()(return_tensor)

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### SVG Labeling Interface")
    with gr.Row():
        with gr.Column():
            curr_description_display = gr.Textbox(label="Description", value=df.loc[0, 'description'], interactive=False)
            index_input = gr.Number(label="SVG Index", value=0, interactive=True)
            curr_label_display = gr.Textbox(label="Current Label", value=df.loc[0, 'manual_label'], interactive=False)
            get_new_random_svg_button = gr.Button("Get New Random SVG")
        svg_display = gr.Image()
    with gr.Row():
        good_button = gr.Button("Good")
        default_box_button = gr.Button("Default Box")
        bad_button = gr.Button("Low Quality")
    save_button = gr.Button("Save DataFrame")
    status = gr.Text()

    # Function to update the display and advance the index
    def update_and_advance(index, label):
        label_svg(index, label)
        new_index = index + 1
        new_svg = show_svg(df.loc[new_index, 'simplified_svg_file_path'])
        curr_label = df.loc[new_index, 'manual_label']
        curr_description = df.loc[new_index, 'description']
        return new_svg, "Label saved! Continue labeling.", new_index, curr_label, curr_description
    
    def update_display(index):
        svg = show_svg(df.loc[index, 'simplified_svg_file_path'])
        curr_label = df.loc[index, 'manual_label']
        curr_description = df.loc[index, 'description']
        return svg, index, curr_label, curr_description

    index_input.change(fn=update_display, inputs=index_input, outputs=[svg_display, index_input, curr_label_display, curr_description_display])
    get_new_random_svg_button.click(fn=lambda: update_display(int(df.sample(1).index[0])), inputs=[], outputs=[svg_display, index_input, curr_label_display, curr_description_display])
    good_button.click(fn=lambda index: update_and_advance(index, "good"), inputs=index_input, outputs=[svg_display, status, index_input, curr_label_display, curr_description_display])
    default_box_button.click(fn=lambda index: update_and_advance(index, "default box"), inputs=index_input, outputs=[svg_display, status, index_input, curr_label_display, curr_description_display])
    bad_button.click(fn=lambda index: update_and_advance(index, "low quality"), inputs=index_input, outputs=[svg_display, status, index_input, curr_label_display, curr_description_display])
    save_button.click(fn=save_df, inputs=[], outputs=status)

demo.launch()
