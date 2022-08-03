import React from "react";
import {
    Typography,
    ListHeader,
    Formula,
    Section,
    Button,
    Action,
    Cycle,
    Space,
    Title,
    List,
    Post,
    Main,
    Get
} from 'react-antd-admin-panel';
import {SectionComponent} from 'react-antd-admin-panel';
import {WarningOutlined} from "@ant-design/icons/lib";
import {message} from "antd";
import {Autocomplete, Conditions, Item} from "react-antd-admin-panel/dist";

export default class Home extends React.Component<any, any> {

    constructor(props) {
        super(props);
        this.state = {
            section: false,
            id: false,
        }
    }

    build() {

        const section = new Section();
        const search = new Autocomplete();
        const result = new Conditions();

        search.key('requestId')
            .label('Find a request')
            .get((value) => new Get()
                .target(() => ({
                    target: `/${value}`,
                }))
            )
            .onChange((object: any) => {
                if (object.value) {
                    result.checkCondition(object);
                    search.tsxSetDisabled(true);
                }
            });

        section.style({ padding: '24px 36px' });
        section.add(new Title().label('Search for a Request').level(1));
        section.add(new Space().top(12));
        section.add(search);
        section.add(result);

        this.setState({ section: section });
    }

    render() {
        return (
            <>{!!this.state.section &&
            <SectionComponent key={this.state.id} main={this.props.main} section={this.state.section}/>}</>
        );
    }

    componentDidMount() {
        this.build()
    }
}