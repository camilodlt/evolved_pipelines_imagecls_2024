using GraphvizDotLang: digraph, node, edge, attr, save

function plot_program(progs::UTCGP.IndividualPrograms, folder::String)
    W = "2"
    H = "2"
    isdir(folder) || mkpath(folder)
    g = digraph("G", rankdir="LR", bgcolor="#f0f0f0", rankstep="2",
            layout="dot",
            ratio="0.4",
            center="2",
            compound="true",
            ordering="in"
        ) |>
        attr(:node; shape="rectangle", style="filled", fillcolor="#ffffff", fontname="Helvetica", fontsize="12", margin="0.1") |>
        attr(:edge, fontname="Helvetica", fontsize="10", color="#333333", constraint="false")
    # g = digraph("G")
    cluster0 = subgraph(g, "cluster_inputs"; label="Inputs", color="#7393B3", style="dashed",
        rank="source")
    cluster1 = subgraph(g, "cluster_prog1";
        label="Program #1", color="#7393B3", style="dashed",
        rank="same")
    cluster2 = subgraph(g, "cluster_prog2";
        label="Program #2", color="#7393B3", style="dashed",
        randk="same"
    )
    all_nodes = []
    input_nodes = []
    n1, n2 = [], []

    # INV edge between last outputs
    last_calling_node_1, last_calling_node_2 = progs[1].program[end].calling_node, progs[2].program[end].calling_node
    x1, x2 = last_calling_node_1.x_position, last_calling_node_2.x_position
    y1, y2 = last_calling_node_1.y_position, last_calling_node_2.y_position
    id1, id2 = "$x1 $y1", "$x2 $y2"

    first_calling_node_1, first_calling_node_2 = progs[1].program[1].calling_node, progs[2].program[1].calling_node
    fx1, fx2 = first_calling_node_1.x_position, first_calling_node_2.x_position
    fy1, fy2 = first_calling_node_1.y_position, first_calling_node_2.y_position
    fid1, fid2 = "$fx1 $fy1", "$fx2 $fy2"

    for (ith_prog, pack) in enumerate(zip(progs.programs[1:2], [cluster1, cluster2], [n1, n2]))
        program, current_g, n = pack
        # r = reverse(program.program)
        for op in program.program
            ### ADD THE CALLING NODE (result) ###
            calling_node = op.calling_node
            x = calling_node.x_position
            y = calling_node.y_position
            id = "$x $y"
            nd = nothing
            group = nothing
            form = "rectangle"
            if (id == id1) || id == id2
                group = "last_nodes"
                form = "hexagon"
            elseif id == fid1 || id == fid2
                group = "first_nodes"
            else
                group = "$ith_prog"
            end
            if calling_node.value isa Number
                nd = node(id; label="$(round(calling_node.value, digits = 2))",
                    group=group,
                    # xlabel=id,
                    shape=form
                )
            else
                tpath = "$folder/$id.png"
                PNGFiles.save(tpath, calling_node.value)
                nd = node(id; shape="box",
                    label="",
                    image=tpath,
                    width=W, height=H,
                    imagescale="true"
                )
            end
            if !(id in all_nodes)
                push!(all_nodes, id)
                push!(n, id)
                current_g |> nd
            end

            ### ADD THE INPUTS ###
            ins = []
            for op_in in op.inputs
                R_node = UTCGP._extract_input_node_from_operationInput(
                    program.program_inputs,
                    op_in,
                )
                node = @unwrap_or R_node throw(ErrorException("Could not extract the input from operation."))
                push!(ins, node)
            end

            for input in ins
                x = input.x_position
                y = input.y_position
                edge_id = "$x $y"
                nd = nothing
                if input.value isa Number
                    nd = node(edge_id; label="$(round(input.value, digits = 2))")
                else
                    tpath = "$folder/$edge_id.png"
                    PNGFiles.save(tpath, input.value)
                    nd = node(edge_id; shape="box",
                        image=tpath, width=W, height=H, imagescale="true")
                end
                if !("$edge_id to $id" in all_nodes)
                    push!(all_nodes, "$edge_id to $id")
                    current_g |> edge(edge_id, id; label=string(op.fn.name),
                        constraint="true",
                        weight="7", minlen="1",
                        #samehead="true" # ugly 
                    )
                end

                if input isa UTCGP.InputNode
                    if !(edge_id in input_nodes)
                        push!(input_nodes, edge_id)
                        cluster0 |> nd
                    end
                else
                    if !(edge_id in all_nodes)
                        push!(all_nodes, edge_id)
                        current_g |> nd
                    end
                end

            end
        end
    end

    # EDGES between input nodes so that they are aligned
    sort!(input_nodes)
    reverse!(input_nodes)
    @show length(input_nodes)
    prev_input = input_nodes[1]
    for next_input in input_nodes[2:end]
        @info "Inputs edge $prev_input, $next_input"
        g |> edge(prev_input, next_input; label="inputnode",
            constraint="true",
            weight="1",
            minlen="0.8",
            style="invis"
        )
        prev_input = next_input
    end

    # EDGES between CALLING NODES of the same program
    # sort!(n1)
    # reverse!(n1)
    @show length(n1)
    prev_input = n1[1]
    for next_input in n1[2:end]
        @info "Inputs edge $prev_input, $next_input"
        g |> edge(prev_input, next_input; label="PROG", constraint="true",
            weight="3",
            style="invis"
        )
        prev_input = next_input
    end
    @show length(n2)
    prev_input = n2[1]
    for next_input in n2[2:end]
        @info "Inputs edge $prev_input, $next_input"
        g |> edge(prev_input, next_input; label="PROG", constraint="true",
            weight="3",
            style="invis"
        )
        prev_input = next_input
    end

    # EDGE CONNECTING THE FIRST TWO 
    g |> edge(
        fid1,
        fid2;
        weight="100000000000000000000",
        minlen="0.01",
        contraint="true",
        label="FIRST",
        style="invis"
    )

    # EDGES BETWEEN THE inputs and FIRST NODE
    for input in input_nodes
        g |> edge(
            input,
            fid1;
            # weight="10",
            # l_tail="cluster_inputs",
            # l_head="cluster_prog1",
            # minlen="2", #e
            contraint="true",
            label="HOLDER",
            style="invis"
        )
        g |> edge(
            input,
            fid2;
            # weight="1",
            # minlen="0.01",e
            contraint="true",
            label="HOLDER",
            style="invis"
        )
    end

    # CONNECT THE LAST TWO
    g |> edge(
        id1,
        id2;
        weight="10000",
        minlen="0.0001",
        contraint="true",
        label="LAST",
        style="invis"
    )


    g, [n1, n2]
end

function eval_plot(x, y, prog, ml, model_arch, shared_inputs, folder, file)
    UTCGP.reset_programs!(prog)
    input_nodes = [
        InputNode(value, pos, pos, model_arch.inputs_types_idx[pos]) for
        (pos, value) in enumerate(x)
    ]
    replace_shared_inputs!(prog, input_nodes)
    outs = UTCGP.evaluate_individual_programs(
        prog,
        model_arch.chromosomes_types,
        ml
    )
    @show outs
    g, nodes = plot_program(prog, folder)
    fn = "$(file)_$(y[1]).png"
    save(g, fn)
end

progs = filter(i -> occursin("pickle", i), readdir("metrics_ga"))
payloads = [
    deserialize("metrics_ga/$i") for i in progs
];


# N images seen per run 
# n_images_per_run = @chain D begin
#     #@subset :filename .== best_on_val[1]
#     @subset :data .== "Val"
#     @groupby :filename
#     @subset :accuracy .== maximum(:accuracy)
#     @groupby :filename
#     @subset :iteration .== minimum(:iteration)
#     @select :iteration :filename :accuracy
#     #@combine :best = maximum(:accuracy)
# end
# [p["best_loss"] for p in payloads] |> argmax # 73
# best_per_run = @chain D begin
#     @subset :data .== "Val"
#     @groupby :filename
#     @combine :best = maximum(:accuracy)
# end
# best_on_val = deepcopy(best_per_run[73, :])
# n_images_per_run[n_images_per_run[:,"filename"] .== best_on_val[1],:] # gens of the best on val

# BEST 50% 
# med = median(best_per_run[:, "best"]) # mid 50% val point
# top50_ids = @chain best_per_run begin
#     @subset :best .>= med
#     @select :filename
# end
# top50_ids = top50_ids[:,1]
# function special_in(values, allowed_values)
#     ret = Bool[]
#     for v in values
#         push!(ret, v in allowed_values)
#     end
#     ret
# end

# test metrics -- 
# # best on val 
# [p["best_loss"] for p in payloads] |> argmax
# test_callback(best_programs[73])
# test_callback = acc_callback(testx, testy, "Test")
# test_acc = []
# for p in best_programs
#     f = test_callback(
#         p
#     )
#     push!(test_acc, f)
# end

