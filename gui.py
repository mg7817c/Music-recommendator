import tkinter as tk
from tkinter import messagebox
from hybrid_recommender import (
    initialise_recommender,
    search_songs,
    recommend_from_song_id,
    TOP_N
)

print("Loading recommender... please wait.")
state = initialise_recommender()
catalog = state["catalog"]
print("Recommender ready.")

current_results = None
selected_song_id = None


def run_search():
    global current_results

    query = search_entry.get().strip()
    matches = search_songs(catalog, query, limit=50)

    current_results = matches.reset_index(drop=True)

    songs_listbox.delete(0, tk.END)

    if current_results.empty:
        songs_listbox.insert(tk.END, "No matching songs found.")
        return

    for _, row in current_results.iterrows():
        songs_listbox.insert(tk.END, f"{row['display']}   [{row['mode']}]")

def select_song():
    global selected_song_id

    selection = songs_listbox.curselection()
    if not selection:
        messagebox.showwarning("No selection", "Please select a song first.")
        return

    idx = selection[0]

    if current_results is None or current_results.empty:
        messagebox.showwarning("No results", "Please search for a song first.")
        return

    row = current_results.iloc[idx]
    selected_song_id = row["song_id"]

    selected_song_label.config(
        text=f"Selected song: {row['display']}   [{row['mode']}]"
    )
    mode_label.config(text="Recommendation mode: ")
    recommendations_listbox.delete(0, tk.END)


def recommend_next():
    if not selected_song_id:
        messagebox.showwarning("No song selected", "Please search and select a song first.")
        return

    mode, recommendations = recommend_from_song_id(
        song_id=selected_song_id,
        shared_songs=state["shared_songs"],
        playlist_only=state["playlist_only"],
        audio_only=state["audio_only"],
        cf_similarity_df=state["cf_similarity_df"],
        cbf_similarity_df=state["cbf_similarity_df"],
        popularity_df=state["popularity_df"],
        n=TOP_N
    )

    mode_label.config(text=f"Recommendation mode: {mode}")
    recommendations_listbox.delete(0, tk.END)

    if not recommendations:
        recommendations_listbox.insert(tk.END, "No recommendations found.")
        return

    for song in recommendations:
        recommendations_listbox.insert(tk.END, song.title())


root = tk.Tk()
root.title("Hybrid Music Recommender")
root.geometry("980x680")

title_label = tk.Label(root, text="Hybrid Music Recommender", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

instructions_label = tk.Label(
    root,
    text="Search for a song, select it, then get next-song recommendations.",
    font=("Arial", 11)
)
instructions_label.pack(pady=5)

search_frame = tk.Frame(root)
search_frame.pack(pady=10)

search_entry = tk.Entry(search_frame, width=50, font=("Arial", 11))
search_entry.grid(row=0, column=0, padx=5)

search_button = tk.Button(search_frame, text="Search", width=15, command=run_search)
search_button.grid(row=0, column=1, padx=5)

songs_label = tk.Label(root, text="Matching Songs", font=("Arial", 12, "bold"))
songs_label.pack()

songs_listbox = tk.Listbox(root, width=95, height=15, font=("Arial", 10))
songs_listbox.pack(pady=8)

select_button = tk.Button(root, text="Select Song", width=20, command=select_song)
select_button.pack(pady=5)

selected_song_label = tk.Label(root, text="Selected song: None", font=("Arial", 11))
selected_song_label.pack(pady=8)

recommend_button = tk.Button(root, text="Recommend Next Songs", width=25, command=recommend_next)
recommend_button.pack(pady=10)

mode_label = tk.Label(root, text="Recommendation mode: ", font=("Arial", 11, "italic"))
mode_label.pack(pady=5)

recommendations_label = tk.Label(root, text="Recommendations", font=("Arial", 12, "bold"))
recommendations_label.pack()

recommendations_listbox = tk.Listbox(root, width=95, height=12, font=("Arial", 10))
recommendations_listbox.pack(pady=8)

root.mainloop()