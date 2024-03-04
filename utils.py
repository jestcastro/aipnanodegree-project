class CNNArch:
    VGG16 = "vgg16"
    RESNET = "resnet18"


def print_validation_progress(
    headers_already_printed,
    epoch,
    epochs,
    running_loss,
    print_every,
    validation_loss,
    accuracy,
    dataloader,
):
    if not headers_already_printed:
        header = f"{'Epoch':<8} | {'Train Loss':<12} | {'Validation Loss':<17} | {'Validation Accuracy':<19} "
        print(header)
        print("-" * len(header))

    epochs_count = f"{epoch+1}/{epochs}"
    epochs_to_print = f"{epochs_count:<8}"
    train_loss_to_print = f"{running_loss / print_every:<12.5f}"
    validation_loss_to_print = f"{validation_loss / len(dataloader):<17.5f}"
    accuracy_to_print = f"{accuracy / len(dataloader):<19.5f}"
    print(
        f"{epochs_to_print} | {train_loss_to_print} | {validation_loss_to_print} | {accuracy_to_print}"
    )
    return True


def print_test_progress(test_loss, accuracy, dataloader):
    header = f"{'Test Loss':<9} | {'Test Accuracy':<12}"
    print(header)
    print("-" * len(header))
    test_loss_to_print = f"{test_loss / len(dataloader):<9.5f}"
    accuracy_to_print = f"{accuracy / len(dataloader):<12.5f}"
    print(f"{test_loss_to_print} | {accuracy_to_print} ")


def print_results(cat_to_name, top_probabilities, top_classes):
    longest_label = max(len(cat_to_name[label]) for label in top_classes)
    header = f"{'Category':<{longest_label}} | {'Probability':<12}"
    print(header)
    print("-" * len(header))
    for i in range(len(top_classes)):
        label = cat_to_name[top_classes[i]]
        probability = top_probabilities[i]
        print(f"{label:<{longest_label}} | {probability * 100:<12.2f}")
