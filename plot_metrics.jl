using JSON3
using DataFrames
using DataFramesMeta
using Makie
using Statistics
using UnfoldMakie
using CSV
# using PlotlyJS
using MathTeXEngine

# PAR PLOT 
confs = CSV.read("paper_folder/sup/confs_irace.csv", DataFrame
)
color = :batlow
colors = cgrad(color, 10, categorical=true)
function multiplex_id(id)
    if id == 1
        "Second Best"
    elseif id == 30
        "Third Best"
    elseif id == 68
        "Best"
    else
        "Other"
    end
end

function multiplex_color(id)
    if id == 1
        colors[2]
    elseif id == 30
        colors[3]
    elseif id == 68
        colors[6]
    else
        colors[1]
    end
end

confs = confs[:, 1:end-1]
ncols = ncol(confs)
select!(confs, ".ID.", "acc_w", :)
x_pos = collect(1:ncols)

confs = confs[:, x_pos]
ax_labels = names(confs)
confs[:, "Ranking"] .= [multiplex_id(id) for id in confs[:, ".ID."]]
confs[:, "Rcol"] .= [i == "Other" ? 0.1 : 4.1 for i in confs[:, "Ranking"]]
ncols = ncol(confs)
ax_labels = ["ρ", "μ", "n", "e", "new", "t", "bs", "reps"]

dims = []
limits = [
    [0.0, 1.0],
    [1, 4],
    [10, 60],
    [2, 100],
    [2, 1000],
    [2, 20],
    [5, 2000],
    [2, 100]]
ticks =
    [
        floor.(LinRange(limits[1][1], limits[1][2], 8), digits=2), # rho
        [1, 2, 3, 4], # mu
        [10, 20, 30, 40, 50, 60], # n
        round.(Int, LinRange(limits[4][1], limits[4][2], 7)), # elite
        round.(Int, LinRange(limits[5][1], limits[5][2], 7)), # new
        round.(Int, LinRange(limits[6][1], limits[6][2], 5)), # t
        round.(Int, LinRange(limits[7][1], limits[7][2], 10)), # bs
        round.(Int, LinRange(limits[8][1], limits[8][2], 10))
    ]

@assert length(ticks) == length(ax_labels)
for (i, i_col) in enumerate(2:9)
    l = limits[i]
    n = ax_labels[i]
    y = collect(confs[:, i_col])
    how_many = length(unique(y))
    ts = ticks[i]
    @show n l ts
    a = attr(
        range=[l[1], l[2]],
        label=n,
        tickvals=ts,
        values=confs[:, i_col])
    push!(dims, a)
end
# cols = [[id, "#" * hex(RGB(multiplex_color(id)))] for id in confs[:, ".ID."]]
mytrace = parcoords(;
    line=attr(
        #autocolorscale=false, cauto=false,
        cmin=0.1, cmax=4.1,
        color=confs[:, "Rcol"] |> collect,
        colorscale=[
            [0, "rgb(168, 221, 230)"],
            [0.1, "rgb(168, 221, 230)"],
            [0.2, "rgb(168, 221, 230)"],
            [0.3, "rgb(168, 221, 230)"],
            [0.4, "rgb(168, 221, 230)"],
            [0.5, "rgb(168, 221, 230)"],
            [0.51, "rgb(0,0,0)"],
            [0.9, "rgb(0,0,0)"],
            [1, "rgb(0,0,0)"]
        ],
        showscale=true,
        colorbar=attr(
            tickvals=[0.1, 4.1],
            ticktext=["Other", "Elite"],
            len=0.3,
            orientation='h', tickmode="array", side="top"
        )
    ),
    dimensions=dims,
    labelfont=attr(size=37, style="italic", lineposition="under+over", shape="auto"),
    tickfont=attr(size=25))

layout = Layout(
    title="",
    xaxis_title="",
    yaxis_title="",
    legend_title="",
    font=attr(
        family="Computer Modern",
        size=32,
        color="black"
    ),
    margin=attr(
        l=90, r=90
    ))

p = PlotlyJS.plot(mytrace, layout)
PlotlyJS.savefig(PlotlyJS.plot(mytrace, layout), "paper_folder/assets/fig_irace1.png",
    width=trunc(Int, 900 * 1.1), height=trunc(Int, 400),
    scale=1)

# BOXPLOT PLOT
results = CSV.read("paper_folder/sup/results_irace.csv", DataFrame, type=Float64)
mean_per_conf = mean.(skipmissing.(eachcol(results)))
results_asc_order = results[:, cat]
results_asc_order_long = stack(results_asc_order, 1:ncol(results_asc_order))
values = results_asc_order_long[.!(ismissing.(results_asc_order_long[:, 2])), 2]
categories = results_asc_order_long[.!(ismissing.(results_asc_order_long[:, 2])), 1]
ordered_categories = []
prev = categories[1]
index = 1
for true_cat in categories
    if true_cat == prev
        println("index : $true_cat at index $index")
        push!(ordered_categories, index)
    else
        index = index + 1
        prev = true_cat
        println("New index : $true_cat at index $index")
        push!(ordered_categories, index)
    end
end

textheme = Theme(fonts=(; regular=texfont(:text),
    bold=texfont(:bold),
    italic=texfont(:italic),
    bold_italic=texfont(:bolditalic)))

final_p = with_theme(textheme) do
    f = Figure(size=(900, 400), fontsize=20)
    ax = Axis(f[1, 1],
        title="",
        xlabel="Configurations tried by Irace",
        ylabel="Negative Validation Accuracy (minimized)",
        xticklabelsvisible=false,
        xticksvisible=false,
        xgridvisible=false,
        yticklabelsize=15,
        # xlabelfont=:normal,
        # ylabelfont=:normal,
        yticks=WilkinsonTicks(4; k_min=4, k_max=10)
    )
    boxplot!(ordered_categories, identity.(values),
        show_outliers=false,
        color=cgrad(color, 10, categorical=true)[4],
        gap=0.7,
        whiskercolor=cgrad(color, 10, categorical=true)[6],
        width=2.5,
        medianlinewidth=3
    )
    f
end
save("paper_folder/assets/fig_irace2.png", final_p)

# METRICS PLOT # 
function read_folder(f::String)
    folder = readdir(f)
    ds = []
    for file in folder
        if occursin("json", file)
            @info "Reading file : $file"
            path = joinpath(pwd(), f, file)
            d = read_file(path)
            d[!, "filename"] .= file
            push!(ds, d)
        end
    end
    vcat(ds..., cols=:union)
end

function read_file(f::String)
    lines = readlines(f)
    lines_JSON = JSON3.read.(lines)
    columns_metrics = keys(lines_JSON[1]) |> collect
    columns_settings = keys(lines_JSON[end]["params"]) |> collect
    cols = vcat(columns_metrics, columns_settings)
    store = Dict([c => [] for c in cols]...)
    for line in lines_JSON[begin:end-1]
        for col_name in columns_metrics
            push!(store[col_name], line[col_name])
        end
    end
    for col_name in columns_settings
        for i in 1:length(lines_JSON[begin:end-1])
            push!(store[col_name], lines_JSON[end][:params][col_name])
        end
    end
    DataFrame(store)
end

D = read_folder("metrics_ga_hed")

f = Figure()
ax = Axis(f[1, 1],
    title="",
    xlabel="Generations",
    ylabel="Mean Accuracy",
    yticks=WilkinsonTicks(12; k_min=12, k_max=20)
)
unique_files = unique(D[!, "filename"])

# GENS AVG 
sub = @chain D begin
    @groupby :data :iteration
    @combine :acc_mean = mean(:accuracy) :acc_std = std(:accuracy) :acc_min = minimum(:accuracy) :acc_max = maximum(:accuracy)
    @transform :low_acc = :acc_mean - :acc_std :high_acc = :acc_mean + :acc_std
    @transform :data_seen = :iteration * (3 * 100)
end
L = []
x = identity.(sub[sub[!, "data"].=="Train", "iteration"])
o = lines!(ax,
    x,
    sub[sub[!, "data"].=="Train", "acc_mean"];
    linestyle=:dash,
    color=colors[1]
)
push!(L, o)
o = lines!(ax,
    x,
    sub[sub[!, "data"].=="Val", "acc_mean"];
    linestyle=:solid,
    color=colors[2]
)
push!(L, o)
btrain = band!(
    x,
    sub[sub[!, "data"].=="Val", "low_acc"],
    sub[sub[!, "data"].=="Val", "high_acc"];
    alpha=0.2, color=colors[1]
)
bval = band!(
    x,
    sub[sub[!, "data"].=="Train", "low_acc"],
    sub[sub[!, "data"].=="Train", "high_acc"];
    alpha=0.2, color=colors[2]
)
# push!(L, o)
# Legend(f[1, 2],
#     [[L[1], btrain], [L[2], bval]],
#     ["Train Avg (100 runs)", "Val Avg (100 runs)"])
Legend(f[1, 2],
    [L[1], L[2]],
    ["Train Avg (100 runs)", "Val Avg (100 runs)"])

# Legend(f[1, 3],
#     [btrain, bval],
#     ["Train Avg (100 runs)", "Val Avg (100 runs)"])

# EVAL 
# L = []
# for filename in unique_files
#     sub = @chain D begin
#         @subset(:filename .== filename, :data .== "Val")
#     end
#     cost_per_iter = (sub[!, "n_new"][1] + sub[!, "n_elite"][1]) * sub[!, "n_train"][1]
#     iters = nrow(sub)
#     total_evals = cost_per_iter * iters
#     accs = Vector{Float64}(undef, total_evals)
#     last = 1
#     for i in 1:iters
#         accs[last:last+cost_per_iter-1] .= sub[!, "accuracy"][i]
#         last = last + cost_per_iter
#     end
#     o = lines!(ax, collect(1:total_evals), accs)
#     push!(L, o)
# end

# for filename in unique_files
#     sub = @chain D begin
#         @subset(:filename .== filename, :data .== "Train")
#     end
#     cost_per_iter = (sub[!, "n_new"][1] + sub[!, "n_elite"][1]) * sub[!, "n_train"][1]
#     iters = nrow(sub)
#     total_evals = cost_per_iter * iters
#     accs = Vector{Float64}(undef, total_evals)
#     last = 1
#     for i in 1:iters
#         accs[last:last+cost_per_iter-1] .= sub[!, "accuracy"][i]
#         last = last + cost_per_iter
#     end
#     o = lines!(ax, collect(1:total_evals), accs)
#     push!(L, o)
# end

# Legend(f[1, 2],
#     [L[1], L[2]],
#     ["Best Test", "Best Train"])

# for filename in unique_files
#     sub = @chain D begin
#         @subset(:filename .== filename, :data .== "Train")
#     end
#     cost_per_iter = (sub[!, "n_new"][1] + 1) * sub[!, "n_train"][1]
#     iters = nrow(sub)
#     _to_add = sub[!, "n_new"][1] * sub[!, "n_train"][1]
#     total_evals = _to_add + (cost_per_iter * (iters - 1))
#     accs = [] #Vector{Float64}(undef, total_evals)
#     last = 1
#     for i in 1:iters
#         if i == 1
#             # @show last:last+_to_add
#             # accs[last:last+_to_add] .= sub[!, "accuracy"][i]
#             # last = last + _to_add
#             push!(accs, [sub[!, "accuracy"][i] for x in 1:_to_add]...)
#         else
#             # @show (last+1):last+cost_per_iter
#             # accs[(last+1):last+cost_per_iter] .= sub[!, "accuracy"][i]
#             # last = last + cost_per_iter
#             push!(accs, [sub[!, "accuracy"][i] for x in 1:cost_per_iter]...)
#         end
#     end
#     # println(mean(accs))
#     # break
#     lines!(ax, collect(1:total_evals), deepcopy(accs))
# end

# function test_score(mach, datax, datay)
#     e_test = Ensemble(acc_callback(datax, datay, "Test"))
#     outs = e_test(PopulationPrograms(best_programs), model_arch, ml)
#     @show outs[5] |> mean
#     df_x1 = DataFrame(map(x -> map(y -> y[1], x), outs[1]), ["gp_1_$i" for i in 1:length(best_programs)])
#     df_x2 = DataFrame(map(x -> map(y -> y[2], x), outs[1]), ["gp_2_$i" for i in 1:length(best_programs)])
#     df_x = hcat(df_x1, df_x2)
#     df_y = DataFrame([[i[1] == 2 ? 1 : 0 for i in datay]], ["y"])
#     ps = predict(mach, df_x)
#     @info [(pred > 0.1 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.2 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.3 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.4 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.5 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.6 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
# end
# function test_score(mach, outs)
#     # e_test = Ensemble(acc_callback(datax, datay, "Test"))
#     # outs = e_test(PopulationPrograms(best_programs), model_arch, ml)
#     @show outs[5] |> mean
#     df_x1 = DataFrame(map(x -> map(y -> y[1], x), outs[1]), ["gp_1_$i" for i in 1:length(best_programs)])
#     df_x2 = DataFrame(map(x -> map(y -> y[2], x), outs[1]), ["gp_2_$i" for i in 1:length(best_programs)])
#     df_x = hcat(df_x1, df_x2)
#     df_y = DataFrame([[i[1] == 2 ? 1 : 0 for i in datay]], ["y"])
#     ps = predict(mach, df_x)
#     @info [(pred > 0.1 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.2 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.3 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.4 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.5 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
#     @info [(pred > 0.6 ? 1 : 0) == gt ? 1 : 0 for (gt, pred) in zip(df_y[:, 1], pdf.(ps, 1))] |> mean
# end


# READ DL DATA 
df_all = DataFrame()
files = glob("paper_folder/assets/*/*/cnn_metrics*.csv")
for file in files
    fn = basename(file)
    model = basename(dirname(dirname(file)))
    df = CSV.read(file, DataFrame)
    df[!, "filename"] .= fn
    df[!, "model"] .= model
    df[!, "Color Space"] .= split(split(file, ".csv")[1], "_")[end]
    append!(df_all, df)
end

color = :Dark2_8
colors_m = cgrad(color, 6, categorical=true)
colors_m = [
    RGB(43 / 255, 131 / 255, 186 / 255),
    RGB(171 / 255, 221 / 255, 164 / 255),
    RGB(253 / 255, 174 / 255, 97 / 255),
    RGB(215 / 255, 25 / 255, 28 / 255),
]

final_p = with_theme(textheme) do
    # PLOT WITH DL 
    f = Figure(size=(1000, 500), fontsize=17, figure_padding=25)
    ax = Axis(f[1, 1],
        title="",
        xlabel="Number of images seen",
        ylabel="Accuracy",
        yticks=WilkinsonTicks(12; k_min=12, k_max=20),
        xticks=[262144 * i for i in 1:5],
        xtickformat=values -> [L"%$(round(value / 1000000, digits = 3))M" for value in values],
        xlabelsize=20,
        ylabelsize=20
    )

    # PLOT CGP

    # -- LINES
    o = lines!(ax,
        sub[sub[!, "data"].=="Train", "data_seen"],
        sub[sub[!, "data"].=="Train", "acc_mean"];
        linestyle=:solid, linewidth=2,
        color=colors_m[1]
    )
    o = lines!(ax,
        sub[sub[!, "data"].=="Val", "data_seen"],
        sub[sub[!, "data"].=="Val", "acc_mean"];
        linestyle=:solid,
        linewidth=3,
        color=colors_m[2]
    )

    # -- MAX LINES
    o = lines!(ax,
        sub[sub[!, "data"].=="Train", "data_seen"],
        sub[sub[!, "data"].=="Train", "acc_max"];
        linestyle=:dot,
        color=colors_m[1], alpha=0.7
    )
    o = lines!(ax,
        sub[sub[!, "data"].=="Val", "data_seen"],
        sub[sub[!, "data"].=="Val", "acc_max"];
        linestyle=:dot, linewidth=2,
        color=colors_m[2],
        alpha=0.8
    )

    # STD BANDS
    # btrain = band!(
    #     identity.(sub[sub[!, "data"].=="Val", "data_seen"]),
    #     sub[sub[!, "data"].=="Val", "low_acc"],
    #     sub[sub[!, "data"].=="Val", "high_acc"];
    #     alpha=0.2, color=colors_m[1]
    # )
    c = c = colors_m[2]
    bval = band!(
        identity.(sub[sub[!, "data"].=="Val", "data_seen"]),
        sub[sub[!, "data"].=="Val", "low_acc"],
        sub[sub[!, "data"].=="Val", "high_acc"];
        alpha=0.5, color=RGBA(c.r, c.g, c.b, 0.4)
    )

    # PLOT DL 
    df_all_sub = @chain df_all begin
        @transform :data_seen = (:epoch) .* 262144
        @subset :data_seen .<= 2000000
    end

    Cs = [
        ["rgbhsvhed"], # CNN best on val is rgb 0.84
        ["rgbhsvhed"] # Resnet best on val is 0.823
    ]
    Ms = ["DLmodels", "DLmodels_resnet18"]
    Ms_names = ["Vanilla CNN", "Resnet18"]
    col_col = "Color Space"
    model_col = "model"
    for (i, model) in enumerate(Ms)
        model_name = Ms_names[i]
        # col = colors_m[2+i]
        colors_m_to_plot_for_model = Cs[i]
        for (ith_color, color_space) in enumerate(colors_m_to_plot_for_model)
            ci = 2 + i + ith_color - 1
            col = colors_m[ci]
            @show col
            println("Plotting lines for model : $model_name ($model) in Cs $color_space")
            # train
            x = @chain df_all_sub begin
                @subset $model_col .== model $col_col .== color_space
                @select :data_seen
            end
            y = @chain df_all_sub begin
                @subset $model_col .== model $col_col .== color_space
                @select :accuracy
            end
            o = lines!(ax, x[:, 1], y[:, 1],
                linestyle=(:dash, :loose),
                linewidth=2,
                color=col
            )
            #val
            y = @chain df_all_sub begin
                @subset $model_col .== model $col_col .== color_space
                @select :val_accuracy
            end
            o = lines!(ax, x[:, 1], y[:, 1],
                linestyle=:solid,
                linewidth=3,
                color=col
            )
        end
    end

    style_lines = [
        LineElement(linestyle=:solid, color=colors_m[2]), # val
        LineElement(linestyle=:solid, color=colors_m[1]), # train
    ]
    style_lines_dl = [
        LineElement(linestyle=:solid, color=:black), # val
        LineElement(linestyle=:dash, color=:black), # train
    ]
    max_lines = [
        LineElement(linestyle=:dot, color=colors_m[2]), # val
        LineElement(linestyle=:dot, color=colors_m[1]), # train
    ]
    color_space = [
        [
            LineElement(linestyle=:solid, color=colors_m[3], points=Point2f[(0.0, 0.8), (1, 0.8)]),
            LineElement(linestyle=:dash, color=colors_m[3], points=Point2f[(0.0, 0.2), (1, 0.2)]),
        ], # vanilla Gray
        [
            LineElement(linestyle=:solid, color=colors_m[4], points=Point2f[(0.0, 0.8), (1, 0.8)]),
            LineElement(linestyle=:dash, color=colors_m[4], points=Point2f[(0.0, 0.2), (1, 0.2)]),
        ],
        # [
        #     LineElement(linestyle=:solid, color=colors_m[5], points=Point2f[(0.0, 0.8), (1, 0.8)]),
        #     LineElement(linestyle=:dash, color=colors_m[5], points=Point2f[(0.0, 0.2), (1, 0.2)]),
        # ],
        # [
        #     LineElement(linestyle=:solid, color=colors_m[6], points=Point2f[(0.0, 0.8), (1, 0.8)]),
        #     LineElement(linestyle=:dash, color=colors_m[6], points=Point2f[(0.0, 0.2), (1, 0.2)]),
        # ], # vanilla Gray
        # [
        #     LineElement(linestyle=:solid, color=colors_m[7], points=Point2f[(0.0, 0.8), (1, 0.8)]),
        #     LineElement(linestyle=:dash, color=colors_m[7], points=Point2f[(0.0, 0.2), (1, 0.2)]),
        # ],
        # [
        #     LineElement(linestyle=:solid, color=colors_m[8], points=Point2f[(0.0, 0.8), (1, 0.8)]),
        #     LineElement(linestyle=:dash, color=colors_m[8], points=Point2f[(0.0, 0.2), (1, 0.2)]),
        # ],
    ]

    # ga_lines = [
    #     LineElement(linestyle=:solid, color=colors_m[1]),
    #     LineElement(linestyle=:solid, color=colors_m[2]),
    # ]
    xlims!(ax, (-20000, 1500000 + 20000))
    axislegend(ax,
        [style_lines, max_lines],
        [["MAGE (Val.)", "MAGE (Train)"], ["MAGE (Val.)", "MAGE (Train)"]],
        ["MAGE Avg.", "MAGE Max."], position=(1, 0),
        groupgap=2, rowgap=1, titlehalign=:left, labelhalign=:left, gridshalign=:left)

    axislegend(ax,
        [style_lines_dl, color_space],
        [["Validation", "Training"], [L"CNN $All^{\ast}$", L"Resnet18 $All^{\ast}$"]],
        ["DL Split", "DL Algorithm"], position=(0.73, 0),
        groupgap=2, rowgap=1, fontsize=10, titlehalign=:left, labelhalign=:left, gridshalign=:left)
    f
end

save("paper_folder/assets/mage_vs_cnn.png", final_p)


# ITERATIONS (DATA SEEN) MAGE
# depeding on the dataset 

D = read_folder("metrics_ga")
unique_files = unique(D[!, "filename"])

function special_in(values, allowed_values)
    ret = Bool[]
    for v in values
        push!(ret, v in allowed_values)
    end
    ret
end

ds = @chain D begin
    @subset :data .== "Val"
    @subset special_in(:filename, unique_files)
    @groupby :filename
    @subset :accuracy .== maximum(:accuracy)
    @groupby :filename
    @subset :iteration .== minimum(:iteration)
    @transform :data_seen = :iteration * (3 * 100)
end
ds[:, "data_seen"] |> mean

# -- top 50 
best_per_run = @chain D begin # filenames sorted by best val acc
    @subset :data .== "Val"
    @groupby :filename
    @combine :best = maximum(:accuracy)
    @orderby -1 * :best
end
top50ids = best_per_run[1:50, 1]
ds = @chain D begin
    @subset :data .== "Val"
    @subset special_in(:filename, top50ids)
    @groupby :filename
    @subset :accuracy .== maximum(:accuracy)
    @groupby :filename
    @subset :iteration .== minimum(:iteration)
    @transform :data_seen = :iteration * (3 * 100)
end
ds[:, "data_seen"] |> mean

# -- top 10
top10ids = best_per_run[1:10, 1]
ds = @chain D begin
    @subset :data .== "Val"
    @subset special_in(:filename, top10ids)
    @groupby :filename
    @subset :accuracy .== maximum(:accuracy)
    @groupby :filename
    @subset :iteration .== minimum(:iteration)
    @transform :data_seen = :iteration * (3 * 100)
end
ds[:, "data_seen"] |> mean

# -- Best 
top1ids = best_per_run[1:1, 1]
ds = @chain D begin
    @subset :data .== "Val"
    @subset special_in(:filename, top1ids)
    @groupby :filename
    @subset :accuracy .== maximum(:accuracy)
    @groupby :filename
    @subset :iteration .== minimum(:iteration)
    @transform :data_seen = :iteration * (3 * 100)
end
ds[:, "data_seen"] |> mean

# -- Most interpretable was 47, the third best : 
# best_genome_73bb6a6b-eff6-40c3-8ccc-3ef6e893be73.pickle

top3ids = best_per_run[3:3, 1]
ds = @chain D begin
    @subset :data .== "Val"
    @subset special_in(:filename, top3ids)
    @groupby :filename
    @subset :accuracy .== maximum(:accuracy)
    @groupby :filename
    @subset :iteration .== minimum(:iteration)
    @transform :data_seen = :iteration * (3 * 100)
end
ds[:, "data_seen"] |> mean

