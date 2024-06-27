from utils import handle_argv
import embedding

if __name__ == "__main__":
    save = True
    mode = "base"
    args = handle_argv("distill_" + mode, "distill.json", mode)
    data, output, labels = embedding.generate_embedding_or_output(
        args=args, output_embed=True, save=save, distill=True
    )
