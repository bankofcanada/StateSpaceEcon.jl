
using Plots
@recipe plot(sd::SimData; nrow=nothing, ncol=nothing, title=:(Default_Title)) = begin

    vars = StateSpaceEcon._names(sd)

    varswithtitle = (title, vars...)

    @series begin
        grid := :none
        showaxis := false
        legend --> false
        # foreground_color_subplot --> :white

        zeros(1)
    end

    # nrow and ncol conditional check 
    if nrow === nothing && ncol === nothing
        nrow, ncol = length(vars), 1
        layout := (nrow + 1, ncol)

        sizex, sizey = 600, nrow * 250
        size   := (sizex, sizey)
    elseif nrow === nothing && ncol isa Number
        nrow = length(vars) / ncol |> ceil |> Int
        layout := (nrow + 1, ncol)

        sizex, sizey = 600, nrow * 200
        size   := (sizex, sizey)
    elseif nrow isa Number && ncol === nothing
        ncol = length(vars) / nrow |> ceil |> Int
        layout := (nrow + 1, ncol)

        sizex, sizey = 600, nrow * 200
        size   := (sizex, sizey)
    else
        layout := (nrow + 1, ncol)

        sizex, sizey = 600, nrow * 200
        size   := (sizex, sizey)
    end
    
    # general settings
    

    for colname in vars
        @series begin
            title  := reshape(map(string, [varswithtitle...]), 1, :)
            titlefont --> font(10, :bold)
            linewidth --> 1.5
            label --> false
            xrotation --> 45
            left_margin --> 4 * Plots.mm

            sd[colname]
        end
    end
end



