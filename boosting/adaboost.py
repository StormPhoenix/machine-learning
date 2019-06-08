from graph import scatter as scatter
import boosting.loader as loader


def main():
    dataMat, labels = loader.loadTestData()
    scatter.showScatterGraph(dataMat, labels)


if __name__ == '__main__':
    main()
