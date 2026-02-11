using LineSearches, LaTeXStrings, Plots

# Pre-Process data for large-scale performance plots
function getSummaryData(functionValData, timeData, fStarData, targetRelAccuracies, maxIter)
    numberSolved = zeros(size(functionValData,1), size(functionValData,2), length(targetRelAccuracies), maxIter)
    times = -ones(size(functionValData,1), size(functionValData,2), length(targetRelAccuracies))

    for i in axes(functionValData,1)
        for j in axes(functionValData,2)
            fStar = fStarData[j]
            data = functionValData[i,j]
            f0 = data[1]
            relErr = (data .- fStar) / (f0 - fStar)
            for (k,acc) in enumerate(targetRelAccuracies)
                l = findfirst(relErr .< acc)  # This will ignore NaNs, as desired
                if !isnothing(l)
                    numberSolved[i,j,k,l:end] .+= 1
                    times[i,j,k] = timeData[i,j][l] - timeData[i,j][1]
                end
            end
        end
    end

    return numberSolved, times
end


# Performance plot with respect to oracle calls
function plotOracle(numberSolved, targetRelAccuracies, methods, problemInstances, metIndices; colors = [])

    p = plot(layout=(1,length(targetRelAccuracies)), size=(1600,500))
    for (plotIdx, acc) in enumerate(targetRelAccuracies)
        accStr = ["Low Accuracy","Medium Accuracy", "High Accuracy"][plotIdx]
        for i in metIndices
            met = methods[i]
            color, style = mapMethodToProperties(met)
            if !isempty(colors)
                color=colors[i]
            end
            plot!(p, vec(sum(numberSolved[i,problemInstances,plotIdx,:], dims=1))/length(problemInstances), subplot=plotIdx, seriestype=:steppost, legend=:none, title=accStr, titlefontsize=16, linewidth=3.5, linestyle = style, color = color)
        end
        plot!(p, ylims=(0.0, 1.01), xlabel="Oracle Calls", subplot=plotIdx)
    end

    return p
end

# Performance plot with respect to real time
function plotTime(times, targetRelAccuracies, methods, problemInstances, metIndices; colors = [], endTime = 60)

    tEnd = endTime

    p = plot(layout=(1,length(targetRelAccuracies)), size=(1600,500))
    for (plotIdx, acc) in enumerate(targetRelAccuracies)
        accStr = ["Low Accuracy","Medium Accuracy", "High Accuracy"][plotIdx]
        for i in metIndices
            met = methods[i]
            ts = times[i,problemInstances,plotIdx]
            ts = ts[ts .>= 0]
            ts = sort(ts)
            ts = vcat(0.0, ts, 2*tEnd) # This cleans up the plot
            color, style = mapMethodToProperties(met)
            if !isempty(colors)
                color=colors[i]
            end
            plot!(p, ts, (1:length(ts))/(length(problemInstances)+1), seriestype=:steppost, subplot=plotIdx, legend=:none, linewidth=3.5, linestyle = style, color=color,title=accStr, titlefontsize=16)
        end
        plot!(p, ylims=(0.0, 1.01),xlims=(0.0, tEnd), xlabel="Time (s)")
    end

    return p
end

function mapProbIdxToEqn(k)
    if k==1
        return "(40)"
    elseif k==2
        return "(38)"
    elseif k==3
        return "(39)"
    elseif k==4
        return "(41)"
    elseif k==5
        return "(42)"
    elseif k==6
        return "(43)"
    end
end

# Performance plots with respect to oracle calls, grouped by problem class
function plotOracle_ByProblemClass(problemTypes, numberSolved, targetRelAccuracies, methods, drawOrder, legendOrder; basepath=[], colors=[])

    groups = [2 3; 1 4; 5 6]
    numDims = 4

    groupNames = ["Regression","Smoothing","Local"]

    layout = @layout [p1; p2; a{1.0w,0.05h}]

    for k=1:3
        i = groups[k,1]
        problemType = problemTypes[i]
        M = Int(size(numberSolved,2)/length(problemTypes)/numDims)
        N = M*length(problemTypes)
        problemInstances = vcat((i-1)*M+1:i*M,  N+(i-1)*M+1:N+i*M, 2N+(i-1)*M+1:2N+i*M, 3N+(i-1)*M+1:3N+i*M)

        p1 = plotOracle(numberSolved, targetRelAccuracies, methods, problemInstances, drawOrder)
        plot!(p1, ylabel="Fraction of Solved Instances of "*mapProbIdxToEqn(i), subplot=1)


        i = groups[k,2]
        problemType = problemTypes[i]
        M = Int(size(numberSolved,2)/length(problemTypes)/numDims)
        N = M*length(problemTypes)
        problemInstances = vcat((i-1)*M+1:i*M,  N+(i-1)*M+1:N+i*M, 2N+(i-1)*M+1:2N+i*M, 3N+(i-1)*M+1:3N+i*M)

        p2 = plotOracle(numberSolved, targetRelAccuracies, methods, problemInstances, drawOrder)
        plot!(p2, ylabel="Fraction of Solved Instances of "*mapProbIdxToEqn(i), subplot=1)

        plot!(p2, title="")

        labels = []
        legendColors = []
        styles = []
        for j in legendOrder
            append!(labels, [methodTitle(methods[j])])
            color, style = mapMethodToProperties(methods[j])
            if !isempty(colors)
                color=colors[i]
            end
            append!(legendColors, [color])
            append!(styles, [style])
        end

        labels = permutedims(labels)
        legendColors = permutedims(legendColors)
        styles = permutedims(styles)

        n = length(labels)

        legend_plot = plot((-n:-1)', (-n:-1)', ylims=(0,1), legendfontsize=9, legendcolumns=Int(ceil(n/2)), legend=:bottom, frame=:none, color=legendColors, labels=labels, ls=styles, linewidth=4)


        pp = plot(p1, p2, legend_plot, layout=layout, bottom_margin = 5Plots.mm, left_margin = 10Plots.mm, size=(1600,900))

        if !isempty(basepath)
            file = basepath*"_"*groupNames[k]*".pdf"
            savefig(file)
        end
        display(pp)
    end
end

# Performance plot with respect to oracle calls and real time
function plotOracleAndTime(numberSolved, times, targetRelAccuracies, methods; drawOrder = [], legendOrder = [], file=[], colors = [], endTime = 60)

    if isempty(drawOrder)
        drawOrder = 1:length(methods)
    end
    if isempty(legendOrder)
        legendOrder = 1:length(methods)
    end

    problemInstances = 1:size(numberSolved,2)

    layout = @layout [blah; a{1.0w,0.05h}]

    p = plot(layout=(2,length(targetRelAccuracies)), size=(1600,900))

    for (plotIdx, acc) in enumerate(targetRelAccuracies)
        accStr = ["Low Accuracy","Medium Accuracy", "High Accuracy"][plotIdx]
        for i in drawOrder
            met = methods[i]
            color, style = mapMethodToProperties(met)
            if !isempty(colors)
                color=colors[i]
            end
            plot!(p, vec(sum(numberSolved[i,problemInstances,plotIdx,:], dims=1))/length(problemInstances), subplot=plotIdx, seriestype=:steppost, label=methodTitle(met), legend=:none, title=accStr, titlefontsize=16, linewidth=3.5, linestyle = style, color = color)
        end
        plot!(p, ylims=(0.0, 1.01), xlabel="Oracle Calls", subplot=plotIdx)
    end
    plot!(p, ylabel="Fraction of Solved Instances", subplot=1)

    tEnd = endTime
    
    for (plotIdx, acc) in enumerate(targetRelAccuracies)
        for i in drawOrder
            met = methods[i]
            ts = times[i,problemInstances,plotIdx]
            ts = ts[ts .>= 0]
            ts = sort(ts)
            ts = vcat(0.0, ts, 2*tEnd) # This cleans up the plot
            color, style = mapMethodToProperties(met)
            if !isempty(colors)
                color=colors[i]
            end
            plot!(p, ts, (0:length(ts)-1)/(length(problemInstances)), seriestype=:steppost, subplot=plotIdx+length(targetRelAccuracies), label=methodTitle(met), legend=:none, linewidth=3.5, linestyle = style, color=color)
        end
        plot!(p, ylims=(0.0, 1.01),xlims=(-1.0, tEnd), xlabel="Time (s)", ylabel="Fraction of Solved Instances", subplot=plotIdx+length(targetRelAccuracies))
    end

    plot!(left_margin = 10Plots.mm)

    labels = []
    legendColors = []
    styles = []
    for i in legendOrder
        append!(labels, [methodTitle(methods[i])])
        color, style = mapMethodToProperties(methods[i])
        if !isempty(colors)
            color = colors[i]
        end
        append!(legendColors, [color])
        append!(styles, [style])
    end

    labels = permutedims(labels)
    legendColors = permutedims(legendColors)
    styles = permutedims(styles)

    n = length(labels)
    legend_plot = plot((-n:-1)', (-n:-1)', ylims=(0,1), legendfontsize=9, legendcolumns=Int(ceil(n/2)), legend=:top, frame=:none, color=legendColors, labels=labels, ls=styles, linewidth=4)


    pp = plot(p, legend_plot, layout=layout)
    plot!(pp, bottom_margin = 5Plots.mm)
    

    display(pp)
end

function mapMethodToProperties(method)
    metType = Symbol(typeof(method))
    if metType == :OBL
        color = :gray59
        style = :dash
    elseif metType == :UFGM
        color = :gray32
        style = :solid
    elseif metType == :AdGD
        color = :lightskyblue
        style = :dash
    elseif metType == :ACFGM
        color = :skyblue3
        style = :solid
    elseif metType == :AdaNAG
        color = :dodgerblue3
        style = :solid
    elseif metType == :NAGF
        color = :mediumpurple1
        style = :dash
    elseif metType == :OSGMB
        color = :mediumpurple3
        style = :solid
    elseif metType == :LBFGS
        if typeof(method.linesearchMode).name.name == :HagerZhang
            color = :coral
            style = :dash
        else
            color = :orangered3
            style = :solid
        end
    elseif metType == :ASPGM11
        color = :aquamarine3
        style = :solid
    elseif metType == :ASPGM
        if method.options.mode == :A
            color = :green
            style = :solid
        else
            color = :green3
            style = :dash
        end
    else
        color = :brown
        style = :solid
    end

    return color, style
end


# Plot comparison of ASPGM with different memory values
function plotMemComparison(solved_Synth, solved_Reg, solved_LP, methods, drawOrder, legendOrder; file=[], colors=[])
    
    layout = @layout [blah; a{1.0w,0.05h}]
    
    if isempty(colors)
        colors = [:teal, :darkgreen, :forestgreen, :limegreen, :aquamarine3]
    end

    p = plot(layout=(1,3), size=(1600,500))
    for i in drawOrder
        met = methods[i]
        color=colors[i]
        style = :solid
        plot!(p, vec(sum(solved_Synth[i,:,1,:], dims=1))/size(solved_Synth,2), subplot=1, seriestype=:steppost, label=methodTitle(met), legend=:none, title="Synthetic Problems", linewidth=3.5, linestyle = style, color = color, xlabel="Oracle Calls")
        plot!(p, vec(sum(solved_Reg[i,:,1,:], dims=1))/size(solved_Reg,2), subplot=2, seriestype=:steppost, label=methodTitle(met), legend=:none, title="Regression Problems", linewidth=3.5, linestyle = style, color = color, xlabel="Oracle Calls")
        plot!(p, vec(sum(solved_LP[i,:,1,:], dims=1))/size(solved_LP,2), subplot=3, seriestype=:steppost, label=methodTitle(met), legend=:none, title="LP Feasibility Problems", linewidth=3.5, linestyle = style, color = color, xlabel="Oracle Calls")
    end
    plot!(p, ylims=(0.0, 1.01))
    plot!(p, ylabel="Fraction of Solved Instances", subplot=1)

    plot!(p,left_margin = 10Plots.mm)

    labels = []
    styles = []
    for i in legendOrder
        append!(labels, [methodTitle(methods[i])])
        append!(styles, [:solid])
    end

    labels = permutedims(labels)
    legendColors = permutedims(colors[legendOrder])
    styles = permutedims(styles)

    n = length(labels)
    legend_plot = plot((-n:-1)', (-n:-1)', ylims=(0,1), legendfontsize=9, legendcolumns=n, legend=:top, frame=:none, color=legendColors, labels=labels, ls=styles, linewidth=4)


    pp = plot(p, legend_plot, layout=layout)
    plot!(pp, bottom_margin = 5Plots.mm)
end


# Plot results from three particular hard problem instances
function plotHardInstances(dataList, fStarList, methods, drawOrder, legendOrder; colors = [])
    layout = @layout [blah; a{1.0w,0.05h}]

    p = plot(layout=(1,3), size=(1600,500))
    for plotIdx=1:3
        data = dataList[plotIdx]
        f0 = data[1,1][1]
        fStar = fStarList[plotIdx]
        str = ["Problem A","Problem B","Problem C"][plotIdx]
        for i in drawOrder
            met = methods[i]
            color, style = mapMethodToProperties(met)
            if !isempty(colors)
                color=colors[i]
            end
            plot!(p, (data[i,1] .- fStar)/(f0 - fStar), subplot=plotIdx, legend=:none, yscale=:log10, title=str, linewidth=4, linestyle = style, color = color)
        end
    end
    plot!(xlims=(-10,500))
    plot!(ylims=(1e-3, 2.0))
    plot!(xlabel="Oracle Calls")
    plot!(ylabel="Scaled Objective Gap", subplot=1)
    
    plot!(left_margin = 10Plots.mm)

    labels = []
    legendColors = []
    styles = []
    for i in legendOrder
        append!(labels, [methodTitle(methods[i])])
        color, style = mapMethodToProperties(methods[i])
        if !isempty(colors)
            color=colors[i]
        end
        append!(legendColors, [color])
        append!(styles, [style])
    end

    labels = permutedims(labels)
    legendColors = permutedims(legendColors)
    styles = permutedims(styles)

    n = length(labels)
    legend_plot = plot((-n:-1)', (-n:-1)', ylims=(0,1), legendfontsize=9, legendcolumns=Int(n/2), legend=:top, frame=:none, color=legendColors, labels=labels, ls=styles, linewidth=4)

    pp = plot(p, legend_plot, layout=layout)
    plot!(pp, bottom_margin = 10Plots.mm)

    display(pp)
end


# Plot detailed results from a single problem instance
function plotDetails(data, methods, xStar, fStar, x0; drawOrder = [], legendOrder = [], colors=[])
    layout = @layout [blah; a{1.0w,0.05h}]
    p = plot(layout=(1,3), size=(1600,500))

    if isempty(drawOrder)
        drawOrder = 1:length(methods)
    end
    if isempty(legendOrder)
        legendOrder = 1:length(methods)
    end
    
    for i in drawOrder
        met = methods[i]
        color, style = mapMethodToProperties(met)
        if !isempty(colors)
            color=colors[i]
        end
        plot!(p, (abs.(data[i,1] .- fStar))/(1/2*norm(xStar - x0)^2), subplot=1, yscale=:log10, legend=:none, linewidth=4, linestyle = style, color = color)
    end
    plot!(xlabel="Oracle Calls", ylims=(1e-16, 1e4), subplot=1)
    plot!(ylabel="Scaled Objective Gap", subplot=1)

    for i in drawOrder
        met = methods[i]
        metType = Symbol(typeof(met))
        color, style = mapMethodToProperties(met)
        if !isempty(colors)
            color=colors[i]
        end
        condition = ((metType == :ASPGM)&&(met.options.mode == :B))||(metType == :OBL)
        if condition
            plot!(p, data[i,3] .+ 1e-16, subplot=2, yscale=:log10, legend=:none, linewidth=4, linestyle = style, color = color)
        end
    end
    plot!(xlabel="Oracle Calls", subplot=2)
    plot!(ylabel="Guarantee", subplot=2)

    for i in drawOrder
        met = methods[i]
        metType = Symbol(typeof(met))
        color, style = mapMethodToProperties(met)
        if !isempty(colors)
            color=colors[i]
        end
        if ((metType == :ASPGM)&&(met.options.mode == :B))||(metType == :OBL)

            if metType == :OBL
                plot!(p, vcat(0.0,data[i,4]) .+ 1e-16, subplot=3, legend=:none, linewidth=4, linestyle = style, color = color, yscale=:log10)
            else 
                plot!(p, data[i,4] .+ 1e-16, subplot=3, legend=:none, linewidth=4, linestyle = style, color = color, yscale=:log10)
            end
        end
    end
    plot!(ylabel="Accumulated Error "*latexstring("\\Delta_n"), subplot=3)
    plot!(xlabel="Oracle Calls", subplot=3)
    plot!(bottom_margin = 10Plots.mm)
    plot!(left_margin = 10Plots.mm)
    plot!(ylims=(1e-16, 1e4), subplot=3)

    labels = []
    legendColors = []
    styles = []
    for i in legendOrder
        append!(labels, [methodTitle(methods[i])])
        color, style = mapMethodToProperties(methods[i])
        append!(legendColors, [color])
        append!(styles, [style])
    end

    labels = permutedims(labels)
    legendColors = permutedims(legendColors)
    styles = permutedims(styles)

    n = length(labels)
    legend_plot = plot((-n:-1)', (-n:-1)', ylims=(0,1), legendfontsize=9, legendcolumns=Int(ceil(n/2)), legend=:top, frame=:none, color=legendColors, labels=labels, ls=styles, linewidth=4)

    pp = plot(p, legend_plot, layout=layout)
    plot!(pp, bottom_margin = 10Plots.mm)

end

# Plot oracle and real-time results for a single problem instance
function plotSingleInstance(functionValData, timeData, methods, fStar; endTime = 30, oracleCalls = 500, drawOrder = [], legendOrder = [], colors=[])
    layout = @layout [blah; a{1.0w,0.05h}]
    p = plot(layout=(1,2), size=(1600,500))

    if isempty(drawOrder)
        drawOrder = 1:length(methods)
    end
    if isempty(legendOrder)
        legendOrder = 1:length(methods)
    end
    
    for i in drawOrder
        f0 = functionValData[i][1]
        met = methods[i]
        color, style = mapMethodToProperties(met)
        if !isempty(colors)
            color=colors[i]
        end
        plot!(p, (abs.(functionValData[i][1:min(length(functionValData[i]), oracleCalls+1)] .- fStar))/(f0 - fStar), subplot=1, yscale=:log10, legend=:none, linewidth=4, linestyle = style, color = color)
    end
    plot!(xlabel="Oracle Calls", ylims=(1e-16, 1e4), subplot=1)
    plot!(ylabel="Scaled Objective Gap", subplot=1)

    for i in drawOrder
        f0 = functionValData[i][1]
        t0 = timeData[i][1]
        met = methods[i]
        color, style = mapMethodToProperties(met)
        if !isempty(colors)
            color=colors[i]
        end
        ts = timeData[i] .- t0
        plot!(p, ts[ts .<= endTime], (abs.(functionValData[i][ts .<= endTime] .- fStar))/(f0 - fStar), subplot=2, yscale=:log10, legend=:none, linewidth=4, linestyle = style, color = color)
    end
    plot!(xlabel="Real Time", ylims=(1e-16, 1e4), subplot=2)
    plot!(ylabel="Scaled Objective Gap", subplot=2)


    plot!(bottom_margin = 10Plots.mm)
    plot!(left_margin = 10Plots.mm)
    plot!(ylims=(1e-16, 1e4), subplot=3)

    labels = []
    legendColors = []
    styles = []
    for i in legendOrder
        append!(labels, [methodTitle(methods[i])])
        color, style = mapMethodToProperties(methods[i])
        if !isempty(colors)
            color=colors[i]
        end
        append!(legendColors, [color])
        append!(styles, [style])
    end

    labels = permutedims(labels)
    legendColors = permutedims(legendColors)
    styles = permutedims(styles)

    n = length(labels)
    legend_plot = plot((-n:-1)', (-n:-1)', ylims=(0,1), legendfontsize=9, legendcolumns=Int(ceil(n/2)), legend=:top, frame=:none, color=legendColors, labels=labels, ls=styles, linewidth=4)

    pp = plot(p, legend_plot, layout=layout)
    plot!(pp, bottom_margin = 10Plots.mm)

end