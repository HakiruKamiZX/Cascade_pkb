using Serialization
using DataFrames
using LinearAlgebra


#Mean dengan euclidean


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

#menghitung mean dengan cascade
function cascade_classify_mean(data_frame, classifier)
    unique_classes = unique(data_frame[!, end])
    means_by_class = Dict{eltype(unique_classes), Vector{Float64}}()

    for class_label in unique_classes
        class_indices = findall(x -> x == class_label, data_frame[!, end])
        class_data = data_frame[class_indices, 1:end-1]
        
        classified_indices = classifier.(eachrow(class_data))
        classified_data = class_data[classified_indices, :]
        
        means = custom_mean(classified_data)
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
