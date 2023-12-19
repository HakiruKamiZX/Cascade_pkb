using Serialization
using DataFrames
using LinearAlgebra
using Statistics


#Mean dengan euclidean
function euclidean_mean(data_frame)
    means = Float64[]
    for col in eachcol(data_frame[:, 1:end-1])
        col_values = coalesce.(Float64.(col), 0.0)
        n = length(col_values)

        # Menghitung euclidean per kolom
        euclidean_mean_val = mean(col_values)

        if n == 0
            push!(means, NaN)
        else
            push!(means, euclidean_mean_val)
        end
    end
    return means
end


#Membagi data menjadi kelas
function calculate_class_means(data_frame)
    unique_classes = unique(data_frame[!, end])
    means_by_class = Dict{eltype(unique_classes), Vector{Float64}}()

    for class_label in unique_classes
        class_indices = findall(x -> x == class_label, data_frame[!, end])
        class_data = data_frame[class_indices, 1:end-1]
        
        means = euclidean_mean(class_data)
        means_by_class[class_label] = means
    end

    return means_by_class
end


function print_mean_for_class(class_label, means)
    println("Mean for Class $class_label:")
    println(means)
end

function custom_mean(data)
    weights = 1:length(data)
    weighted_sum = sum(data .* weights)
    total_weight = sum(weights)
    return weighted_sum / total_weight
end

#menghitung mean dengan cascade
function cascade_classify_mean(data_frame, classifier)
    unique_classes = unique(data_frame[!, end])
    means_by_class = Dict{eltype(unique_classes), Vector{Float64}}()

    for class_label in unique_classes
        class_indices = findall(x -> x == class_label, data_frame[!, end])
        class_data = data_frame[class_indices, 1:end-1]

        # Apply classifier to each row
        classified_indices = map(classifier, eachrow(class_data))
        classified_data = class_data[classified_indices, :]

        # Calculate mean for the classified data
        means = euclidean_mean(classified_data)
        means_by_class[class_label] = means
    end

    return means_by_class
end

path = "data_9m.mat"

serialized_data = read(path, String)

deserialized_data = deserialize(IOBuffer(serialized_data))

data_matrix = Matrix(deserialized_data)

classes = data_matrix[:, end]

classifier(row) = row[1] < 0 ? true : false

df = DataFrame(data_matrix, :auto)

println("Euclidesn mean result:")
for (class_label, means) in calculate_class_means(df)
    print_mean_for_class(class_label, means)
end

println("\nCascade classify result:")
for (class_label, means) in cascade_classify_mean(df, classifier)
    print_mean_for_class(class_label, means)
end
